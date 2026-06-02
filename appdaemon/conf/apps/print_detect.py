import adbase as ad
from lib.detection_model import *
import cv2
from configparser import ConfigParser
import requests
import yaml
import os
import numpy as np
import base64

class PrintDetect(ad.ADBase):
    '''
    This class is used to detect issues with a 3D print job using the machine learning model. 
    It takes a snapshot of the print job every x seconds (5 by default) and runs the detection model on the image.
    If an issue is detected, notifications are sent and the print is auto-cancelled after a timeout
    unless the user overrides with Continue Print.
    '''
    
    def initialize(self):
        self.cancel_handle = None
        self.alert_active = False
        self.adapi = self.get_ad_api() # get the AppDaemon API
        
        # paths to the model files
        self.model_cfg = "/conf/model/model.cfg"
        self.model_meta = "/conf/model/model.meta"
        self.model_weights = "/conf/model/model-weights-5a6b1be1fa.onnx"
        
        self.warmup_complete = False # flag to check if the printer has warmed up
        
        # load all configuration file variables
        self.load_config()
        self.load_secret_values()
        
        self.printer_status = self.adapi.get_entity(self.printer_status_entity) # get the printer status
        self.print_cameras = [self.adapi.get_entity(e) for e in self.printer_camera_entities] # get all cameras
        self.detected_snapshot_image = "snapshot_0.jpg" # track which camera image triggered detection
        self.detected_annotated_image = None # annotated BGR image for persistent notification
        self.stop_print_button = self.adapi.get_entity(self.printer_stop_button_entity) # get the stop print button
        self.extruder_temp_sensor = self.adapi.get_entity(self.extruder_temp_sensor_entity) # get the extruder temperature sensor
        self.extruder_target_temp_sensor = self.adapi.get_entity(self.extruder_target_temp_sensor_entity) # get the extruder target temperature sensor
        self.net_main_1 = load_net(self.model_cfg, self.model_meta, self.model_weights) # load the ml model
        
        if self.notification_on_warp_up and (self.extruder_temp_sensor is None or self.extruder_target_temp_sensor is None):
            raise RuntimeError("Invalid Config File. ExtruderTempSensor and ExtruderTargetTempSensor must be defined if NotifyOnWarmup is True.")
        
        self.adapi.run_every(self.run_every_c, "now", self.detection_interval) # run the detection every x seconds
        self.adapi.listen_event(self.handle_action, "mobile_app_notification_action") # listen for mobile app notification actions (e.g. stop print or continue)
        
    @staticmethod
    def get_config_value(config: ConfigParser, group: str, id: str, type: type) -> any:
        """
        Get a value from the config file or the default. 

        Args:
            config (ConfigParser): The configuration file parser
            group (str): The group the value belongs to
            id (str): The id of the value to retreive
            type (type): The expected type of the value wanted to be retrieved.

        Raises:
            RuntimeError: Raise error if the retreived type is not the same as the one extected.

        Returns:
            any: The value.
        """
        value = config[group][id] or config['DEFAULT'][id]
        try:
            value = type(value)
            return value
        except ValueError:
            raise RuntimeError(f"Invalid Config File. {group} {id} must be of type {type}.")
        
    def load_secret_values(self) -> None:
        """
        Load the secret values from the secrets.yaml file needed for requesting the camera snapshot.
        """
        secrets_path = os.path.join(os.path.dirname(__file__), '..', 'secrets.yaml')
        with open(secrets_path, 'r') as file:
            secrets = yaml.safe_load(file)
        self.hass_token = secrets.get('HASS_TOKEN')
        self.hass_hostname = secrets.get('HASS_HOSTNAME')
    
    def load_config(self):
        """
        Loads the variables from the config file.
        """
        config = ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), 'config.ini'))
        self.printer_status_entity: str = PrintDetect.get_config_value(config=config, group='printer.entities', 
                                                                id='BinaryIsPrintingSensor', type=str)
        self.printer_printing_state: str = PrintDetect.get_config_value(config=config, group='printer.entities', 
                                                                id='PrintingOnState', type=str)
        if config.has_option('printer.entities', 'PrinterCameras') or config.has_option('DEFAULT', 'PrinterCameras'):
            cameras_str: str = PrintDetect.get_config_value(config=config, group='printer.entities',
                                                            id='PrinterCameras', type=str)
            self.printer_camera_entities = [c.strip() for c in cameras_str.split(',')]
        else:
            single: str = PrintDetect.get_config_value(config=config, group='printer.entities',
                                                        id='PrinterCamera', type=str)
            self.printer_camera_entities = [single]
        self.printer_stop_button_entity: str = PrintDetect.get_config_value(config=config, group='printer.entities', 
                                                                id='PrinterStopButton', type=str)
        self.detection_interval: int = PrintDetect.get_config_value(config=config, group='program.timings', 
                                                                id='RunModelInterval', type=int)
        self.print_termination_time: int = PrintDetect.get_config_value(config=config, group='program.timings', 
                                                                id='TerminationTime', type=int)
        self.detection_threshold: float = PrintDetect.get_config_value(config=config, group='model.detection', 
                                                                id='Threshold', type=float)
        self.detection_nms: float = PrintDetect.get_config_value(config=config, group='model.detection', 
                                                                id='NMS', type=float)
        self.extruder_temp_sensor_entity: str = PrintDetect.get_config_value(config=config, group='notifications.entities', 
                                                                id='ExtruderTempSensor', type=str)
        self.extruder_target_temp_sensor_entity: str = PrintDetect.get_config_value(config=config, group='notifications.entities', 
                                                                id='ExtruderTargetTempSensor', type=str)
        self.notification_on_warp_up: bool = True if PrintDetect.get_config_value(config=config, group='notifications.config',
                                                                id='NotifyOnWarmup', type=str) == 'True' else False
        
    def get_camera_snapshot(self, image_path="snapshot_0.jpg"):
        """
        Get the camera snapshot and decode it into an image.

        Args:
            image_path: The image filename (e.g. snapshot.jpg) to fetch from HA media.

        Returns:
            The decoded image.
        """
        url = f"{self.hass_hostname}/media/local/{image_path}"
        headers = {
            'Authorization': f'Bearer {self.hass_token}'
        }
        response = requests.request("GET", url, headers=headers, data={}, stream=True)
        if response.status_code != 200:
            self.adapi.log(f"Error getting camera snapshot: {response.status_code}")
            return None
        arr = np.asarray(bytearray(response.raw.read()), dtype=np.uint8)
        cv2_img = cv2.imdecode(arr, -1)
        return cv2_img
    
    def draw_annotations(self, image, detections):
        for d in detections:
            name, confidence, (xc, yc, w, h) = d
            x1 = int(xc - w / 2)
            y1 = int(yc - h / 2)
            x2 = int(xc + w / 2)
            y2 = int(yc + h / 2)
            label = f"{name} {confidence:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return image

    def upload_media(self, image_bgr, filename):
        success, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            self.adapi.log(f"Failed to encode {filename}")
            return False
        url = f"{self.hass_hostname}/api/media_source/local_source/upload"
        headers = {"Authorization": f"Bearer {self.hass_token}"}
        resp = requests.post(url, headers=headers, files={"media": (filename, buf.tobytes(), "image/jpeg")})
        if resp.status_code not in (200, 201):
            self.adapi.log(f"Failed to upload {filename}: HTTP {resp.status_code}")
            return False
        self.adapi.log(f"Uploaded {filename} to HA media")
        return True

    def perform_detection(self) -> int:
        """
        Take snapshots from all cameras and run the detection model on each image.
        The camera that detects the most issues determines which snapshot is used
        in the notification.

        Returns:
            int: The number of issues detected. 0 if no issues or snapshots failed.
        """
        max_count = 0
        for i, cam in enumerate(self.print_cameras):
            image_name = f"snapshot_{i}.jpg"
            entity_id = self.printer_camera_entities[i]
            cam.call_service("snapshot", filename=f"/media/{image_name}")
            bgr = self.get_camera_snapshot(image_name)
            if bgr is None:
                self.adapi.log(f"Camera {i} ({entity_id}): snapshot failed, skipping.")
                continue
            detections = detect(self.net_main_1, bgr, thresh=self.detection_threshold, nms=self.detection_nms)
            count = len(detections)
            self.adapi.log(f"Camera {i} ({entity_id}): detected {count} issues")
            if count > max_count:
                max_count = count
                if count > 0:
                    annotated = self.draw_annotations(bgr.copy(), detections)
                    self.detected_annotated_image = annotated
                    annotated_name = f"annotated_{i}.jpg"
                    if self.upload_media(annotated, annotated_name):
                        self.detected_snapshot_image = annotated_name
                    else:
                        self.detected_snapshot_image = image_name
                else:
                    self.detected_snapshot_image = image_name

        self.adapi.log(f"Detection cycle complete: max {max_count} issues across {len(self.print_cameras)} camera(s)")
        return max_count
    
    def send_detection_notification(self):
        """
        Create both a persistent notification (with annotated image)
        and a push notification (for the Android tray with action buttons).
        The print will be auto-cancelled after the timeout unless overridden.
        """
        self.alert_active = True

        # Persistent notification with embedded image
        if self.detected_annotated_image is not None:
            success, buf = cv2.imencode(".jpg", self.detected_annotated_image,
                                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if success:
                b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                img_tag = f"<img src=\"data:image/jpeg;base64,{b64}\" style=\"width:100%;max-width:600px;\">"
            else:
                img_tag = ""
        else:
            img_tag = ""
        self.adapi.call_service("persistent_notification/create",
                                title="3D Print Issue Detected",
                                message=f"An issue with your 3D print has been detected.<br><br>{img_tag}",
                                notification_id="print_detect_alert")

        # Push notification with action buttons
        full_image_url = f"{self.hass_hostname}/media/local/{self.detected_snapshot_image}"
        self.adapi.call_service("notify/notify",
                                message="An issue with your 3D print has been detected. "
                                        f"The print will be cancelled in {self.print_termination_time} seconds if not overridden.",
                                title="3D Print Issue Detected",
                                data={
                                    "image": full_image_url,
                                    "channel": "printer_alarm",
                                    "tag": "print-detect-alert",
                                    "url": "/lovelace/0",
                                    "actions": [
                                        {
                                            "action": "CONTINUE_PRINT",
                                            "title": "Continue Print"
                                        },
                                        {
                                            "action": "CANCEL_PRINT",
                                            "title": "Cancel Print"
                                        }
                                    ],
                                    "push": {
                                        "interruption-level": "critical"
                                    }})
        self.adapi.log(f"Push notification sent with image={full_image_url}")
        self.cancel_handle = self.adapi.run_in(self.cancel_print_callback, self.print_termination_time)

    def notify_on_warmup(self):
        """
        Notify the user when the printer is almost warmed up
        """
        if float(self.extruder_temp_sensor.state) > (0.9 * float(self.extruder_target_temp_sensor.state)) and float(self.extruder_temp_sensor.state) < (0.96 * float(self.extruder_target_temp_sensor.state)) and self.warmup_complete == False:
            self.warmup_complete = True
            self.adapi.call_service("notify/notify", 
                                    message="The 3D printer has almost warmed up. Remove any excess filament before your print starts.", 
                                    title="3D Printer Warming Up",
                                    data={
                                        "image": "/media/local/snapshot_0.jpg"
                                    })
        if float(self.extruder_temp_sensor.state) > (0.96 * float(self.extruder_target_temp_sensor.state)):
            self.warmup_complete = False
        
    def extra_notifications_router(self):
        """
        Check if extra notifications are needed.
        """
        if self.notification_on_warp_up:
            self.notify_on_warmup()
        
    def run_every_c(self, cb_args):
        '''
        This function is called every x seconds to take a snapshot of the print job and run the detection model.
        It will send a notification if an issue is detected.
        '''
        # check if the printer is on and a notification has not already been sent
        if self.printer_status.is_state(self.printer_printing_state) and self.cancel_handle == None:
            # call the extra notifications router to check if any extra notifications are needed
            self.extra_notifications_router()
            # if the printer is on, take a snapshot and run the detection model
            detection_count = self.perform_detection()
            # if an issue is detected, send a notification
            if detection_count > 0:
                self.adapi.log(f"Detection threshold met ({detection_count} issues), sending notification")
                self.send_detection_notification()

    def handle_action(self, event_name, data, kwargs):
        '''
        This is a routing function called when a mobile app notification action is received.
        It will run the appropriate function based on the action received.
        '''
        self.adapi.log(f"Received action: {data}")
        if data["action"] == "CANCEL_PRINT":
            self.cancel_print()
        elif data["action"] == "CONTINUE_PRINT":
            self.continue_print()

    def cancel_print(self):
        '''
        Called when the user taps Cancel Print — stops the print immediately.
        '''
        self.adapi.log("Cancel Print action received — stopping print immediately")
        if self.cancel_handle is not None:
            self.adapi.cancel_timer(self.cancel_handle)
            self.cancel_handle = None
        self.stop_print_button.call_service("press")
        self.adapi.call_service("notify/notify", message="The 3D print has been cancelled.", title="3D Print Cancelled")

    def cancel_print_callback(self, cb_args):
        '''
        Timer callback — auto-cancels the print when the timeout expires.
        '''
        self.adapi.log("Cancel timer expired — auto-cancelling print")
        self.cancel_handle = None
        self.stop_print_button.call_service("press")
        self.adapi.call_service("notify/notify", message="The 3D print has been cancelled.", title="3D Print Cancelled")

    def continue_print(self):
        '''
        Called when the user taps Continue Print — cancels the auto-cancel timer.
        '''
        self.adapi.log("Continue Print action received — cancelling timer, print will continue")
        if self.cancel_handle is not None:
            self.adapi.cancel_timer(self.cancel_handle)
            self.cancel_handle = None
        self.adapi.call_service("notify/notify", message="The 3D print will continue.", title="3D Print Continuing")
        self.alert_active = False
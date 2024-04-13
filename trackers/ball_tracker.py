from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]
        id_name_dict = results.names

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]

            ball_dict[1] = result
        return ball_dict

    # def detect_frame(self, frame):
    #     results = self.model.track(frame, persist=True)[0]
    #     id_name_dict = results.names
    #
    #     player_dict = {}
    #     for box in results.boxes:
    #         track_id = int(box.id.tolist()[0])
    #         result = box.xyxy.tolist()[0]
    #         object_cls_id = box.cls.tolist()[0]
    #         object_cls_name = id_name_dict[object_cls_id]
    #         if object_cls_name == "person":
    #             bbox = result
    #             if track_id not in player_dict:
    #                 player_dict[track_id] = []
    #             player_dict[track_id].append((track_id, bbox))
    #
    #     return player_dict

    # def draw_bboxes(self, video_frames, player_detections):
    #     output_video_frames = []
    #     for frame, player_dict in zip(video_frames, player_detections):
    #         # Draw Bounding Boxes
    #         # print(frame)
    #         for track_id, bbox in player_dict.items():
    #             x1, y1, x2, y2 = bbox
    #             cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    #             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    #         output_video_frames.append(frame)
    #
    #     return output_video_frames

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for i in range(len(video_frames)):
            frame = video_frames[i]
            if i < len(player_detections):
                ball_dict = player_detections[i]
            else:
                ball_dict = {}  # Empty dictionary if no detections for this frame

            # Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball_ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            output_video_frames.append(frame)

        return output_video_frames

    # prdict the between values where ball is not detected
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        df_ball_position = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])  # convert list into dataframe so that we can use interpolation

        #interpolate missing value
        df_ball_position = df_ball_position.interpolate()

        # predit no missing value in starting , so that system do not crash
        df_ball_position = df_ball_position.bfill()
        print(df_ball_position)

        #convert df to list
        ball_positions = [{1:x} for x in df_ball_position.to_numpy().tolist()]

        return ball_positions

    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1,
                                                                                     center=False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()
        df_ball_positions['ball_hit'] = 0
        minimum_change_frame_for_hit = 25

        for i in range(1, len(df_ball_positions) - int(minimum_change_frame_for_hit * 1.2)):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i + 1] < 0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i + 1] > 0

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i + 1, i + int(minimum_change_frame_for_hit * 1.2) + 1):
                    negative_position_change_follow_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0
                    positive_position_change_follow_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

                    if negative_position_change and negative_position_change_follow_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_follow_frame:
                        change_count += 1

                if (change_count > minimum_change_frame_for_hit - 1):
                    df_ball_positions['ball_hit'].iloc[i] = 1
        df_ball_positions[df_ball_positions['ball_hit'] == 1]
        frame_num_ball_hit = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()
        return frame_num_ball_hit

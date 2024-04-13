import cv2

from utils import (read_video, save_video)

from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import Mini_Court

def main():
    input_video_path = 'input_videos/input_video_1.mp4'
    video_frames = read_video(input_video_path)
    # print(video_frames)

    ball_tracker = BallTracker(model_path='models/yolo5_last.pt')

    court_model_path = "models/keypoints_model.pth"
    # court_model_path = "models/model_tennis_court_det.pt"

    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])


    player_tracker = PlayerTracker(model_path='yolov8x')
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=False, stub_path="tracker_stubs/player_detections.pkl")
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=False, stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)


    mini_court = Mini_Court(video_frames[0])

    ball_shot_frame = ball_tracker.get_ball_shot_frames(ball_detections)
    print(ball_shot_frame)

    # player_mini_court_detections = mini_court.convert_bounding_box_to_mini_court_coordinate(player_detections, ball_detections, court_keypoints)

    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    # output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)

    # Draw frame number
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame No. -  {i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2)


    # print(output_video_frames)
    save_video(output_video_frames, "output_videos/output_video.avi")
if __name__ == "__main__":
    main()
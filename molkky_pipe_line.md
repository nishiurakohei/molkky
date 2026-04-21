# 使い方
 1. 依存インストール
pip install mediapipe opencv-python scipy numpy

 2. モデルをダウンロード（初回のみ・約30MB）
python molkky_pose_pipeline.py --download-model

 3. 動画を処理
python molkky_pose_pipeline.py --input throw.mp4

 CSV のみ欲しい場合（高速）
python molkky_pose_pipeline.py --input throw.mp4 --no-video


# 注意
 1. python3.13以降のバージョンでは動かないため,python3.12で動かすのがよい
 2. SSL証明書エラーにぶつかった場合、/Applications/Python\ 3.12/Install\ Certificates.commandを走らせてから実行する

import React, { useEffect, useRef } from "react";

const RealTimeDetection = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    // 啟動 webcam
    navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
      console.log("攝影機啟動成功");
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    }).catch(error => {
      console.error("攝影機啟動失敗", error);
    });

    // 建立 WebSocket
    ws.current = new WebSocket("ws://localhost:8000/ws/detect");
    ws.current.onmessage = (e) => {
      const data = JSON.parse(e.data);
      const img = new Image();
      img.src = `data:image/jpeg;base64,${data.image}`;
      img.onload = () => {
        const canvas = canvasRef.current;
        if (canvas) {
          const ctx = canvas.getContext("2d");
          if (ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);

            // 畫框框
            if (data.detections) {
              data.detections.forEach((det: any) => {
                const [x1, y1, x2, y2] = det.bbox;
                ctx.strokeStyle = "red";
                ctx.lineWidth = 2;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                ctx.fillStyle = "red";
                ctx.font = "16px Arial";
                ctx.fillText(`${det.label} (${(det.confidence * 100).toFixed(1)}%)`, x1 + 4, y1 + 16);
              });
            }
          }
        }
      };
    };

    // 每 200ms 擷取一張 webcam 畫面送到後端
    const interval = setInterval(() => {
      const video = videoRef.current;
      if (!video) return;
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      
      ctx.drawImage(video, 0, 0);
      const dataUrl = canvas.toDataURL("image/jpeg");
      const base64 = dataUrl.split(",")[1];

      console.log("傳送圖片 frame 大小:", base64.length);
      ws.current?.send(base64);
    }, 200);

    return () => {
      clearInterval(interval);
      ws.current?.close();
    };
  }, []);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white">
      <h1 className="text-2xl mb-4">🚏 公車站即時偵測系統</h1>
      <canvas ref={canvasRef} width={640} height={480} className="border-4 border-white rounded-lg shadow-lg" />
      <video ref={videoRef} style={{ display: 'none' }} />
    </div>
  );
};

export default RealTimeDetection;

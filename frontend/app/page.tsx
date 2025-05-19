"use client";
import React, { useEffect, useRef, useState } from "react";

const Page = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ws = useRef<WebSocket | null>(null);

  const [fps, setFps] = useState<number | null>(null);
  const [alert, setAlert] = useState<boolean>(false);
  const [detectionCount, setDetectionCount] = useState<number>(0);

  useEffect(() => {
    // 啟用攝影機
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    });

    // 建立 WebSocket 連線
    ws.current = new WebSocket("ws://localhost:8000/ws/detect");

    ws.current.onmessage = (e) => {
      const data = JSON.parse(e.data);
      const img = new Image();
      img.src = `data:image/jpeg;base64,${data.image}`;

      // 更新 UI 狀態
      setAlert(data.alert);
      setFps(data.inference_time ? Math.round(1 / data.inference_time) : null);
      setDetectionCount(data.detections?.length || 0);

      // 顯示影像與框
      img.onload = () => {
        const canvas = canvasRef.current;
        if (canvas) {
          const ctx = canvas.getContext("2d");
          if (ctx) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0);

            if (data.detections) {
              data.detections.forEach((det: any) => {
                const [x1, y1, x2, y2] = det.bbox;
                ctx.strokeStyle = "red";
                ctx.lineWidth = 2;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                ctx.fillStyle = "red";
                ctx.font = "14px Arial";
                ctx.fillText(`${det.label} (${(det.confidence * 100).toFixed(1)}%)`, x1 + 4, y1 + 16);
              });
            }
          }
        }
      };
    };

    // 定時擷取 webcam 並傳給後端
    const interval = setInterval(() => {
      const video = videoRef.current;
      if (!video) return;

      const canvas = document.createElement("canvas");
      canvas.width = 640;
      canvas.height = 480;
      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.drawImage(video, 0, 0);
        const base64 = canvas.toDataURL("image/jpeg").split(",")[1];
        ws.current?.send(base64);
      }
    }, 200);

    return () => {
      clearInterval(interval);
      ws.current?.close();
    };
  }, []);

  return (
    <div
      className={`flex flex-col items-center justify-center min-h-screen text-white transition-all duration-200 ${
        alert ? "bg-red-900" : "bg-black"
      }`}
    >
      <h1 className="text-2xl mb-2">🚏 公車站即時 AI 偵測系統</h1>

      <div className="text-sm mb-4">
        {fps !== null && <span>FPS: {fps}</span>}
        {alert && <span className="ml-4 text-yellow-300 font-bold">⚠️ 有行動輔具使用者</span>}
        <span className="ml-4">偵測物件數：{detectionCount}</span>
      </div>

      {/* 攝影機畫面 */}
      <video
        ref={videoRef}
        autoPlay
        width={640}
        height={480}
        style={{ border: "2px solid red", display: "block", marginBottom: "10px" }}
      />

      {/* AI 處理後畫面 */}
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        className="border-4 border-white rounded-lg shadow-lg"
      />
    </div>
  );
};

export default Page;

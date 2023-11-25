import { useRef, useEffect } from "react";
import "./App.css";
import Webcam from "react-webcam";
import * as facemesh from "@tensorflow-models/facemesh";
import * as handpose from "@tensorflow-models/handpose";
import * as tf from "@tensorflow/tfjs";
import { drawMesh } from "./util";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const runFacemeshAndHandpose = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });
        if (webcamRef.current) {
          webcamRef.current.video.srcObject = stream;
        }

        const facemeshNet = await facemesh.load({
          inputResolution: { width: 640, height: 480 },
          scale: 0.0,
        });

        const handposeNet = await handpose.load();

        setInterval(() => {
          detect(facemeshNet, handposeNet);
        }, 100);
      } catch (error) {
        console.error("Error accessing camera:", error);
      }
    };

    const detect = async (facemeshNet, handposeNet) => {
      if (
        typeof webcamRef.current !== "undefined" &&
        webcamRef.current !== null &&
        webcamRef.current.video.readyState === 4
      ) {
        const video = webcamRef.current.video;
        const videoWidth = webcamRef.current.video.videoWidth;
        const videoHeight = webcamRef.current.video.videoHeight;

        webcamRef.current.video.width = videoWidth;
        webcamRef.current.video.height = videoHeight;

        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;

        // Detecção de rosto
        const face = await facemeshNet.estimateFaces(video);

        // Detecção de mãos
        const hands = await handposeNet.estimateHands(video);

        const ctx = canvasRef.current.getContext("2d");

        drawMesh(face, ctx);
        drawHands(hands, ctx);
      }
    };

    const drawHands = (hands, ctx) => {
      hands.forEach((hand) => {
        const landmarks = hand.landmarks;
        for (let i = 0; i < landmarks.length; i++) {
          const [x, y] = landmarks[i];
          ctx.beginPath();
          ctx.arc(x, y, 5, 0, 2 * Math.PI);
          ctx.fillStyle = "red";
          ctx.fill();
        }
      });
    };

    runFacemeshAndHandpose();
  }, []); // Empty dependency array ensures this runs only once after the initial render

  return (
    <div
      style={{
        position: "absolute",
        marginLeft: "auto",
        marginRight: "auto",
        left: 0,
        right: 0,

        width: 800, // Adjust for responsive design
        height: "auto", // Adjust for responsive design
      }}
    >
      <Webcam
        ref={webcamRef}
        style={{
          position: "absolute",
          marginLeft: "auto",
          marginRight: "auto",
          left: 0,
          right: 0,
          zIndex: 1,
          width: "100%", // Adjust for responsive design
          height: "auto", // Adjust for responsive design
        }}
      />
      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          marginLeft: "auto",
          marginRight: "auto",
          left: 0,
          right: 0,
          zIndex: 2,
          width: "100%", // Adjust for responsive design
          height: "auto", // Adjust for responsive design
        }}
      />
    </div>
  );
}

export default App;

import { useRef, useEffect, useState, useCallback } from 'react';
import './App.css';
import Webcam from 'react-webcam';
import type { FaceLandmarksDetector } from '@tensorflow-models/face-landmarks-detection';
import type { HandDetector } from '@tensorflow-models/hand-pose-detection';
import '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import { drawFaceMesh, drawHands } from './util';

type LoadingState = {
  faceModel: boolean;
  handModel: boolean;
  camera: boolean;
};

function App() {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number | null>(null);
  const [loading, setLoading] = useState<LoadingState>({
    faceModel: true,
    handModel: true,
    camera: false,
  });
  const [error, setError] = useState<string | null>(null);
  const [faceTrackingEnabled, setFaceTrackingEnabled] = useState<boolean>(true);
  const [handTrackingEnabled, setHandTrackingEnabled] = useState<boolean>(true);
  const [rightHandPosition, setRightHandPosition] = useState<{ 
    x: number; 
    y: number; 
    z: number;
    rotationX: number;
    rotationY: number;
  } | null>(null);

  const detect = useCallback(
    async (
      faceDetector: FaceLandmarksDetector | null,
      handDetector: HandDetector | null
    ) => {
      if (
        !webcamRef.current?.video ||
        webcamRef.current.video.readyState !== 4 ||
        !canvasRef.current
      ) {
        return;
      }

      const video = webcamRef.current.video;
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;

      if (videoWidth === 0 || videoHeight === 0) {
        return;
      }

      video.width = videoWidth;
      video.height = videoHeight;

      const canvas = canvasRef.current;
      canvas.width = videoWidth;
      canvas.height = videoHeight;

      const ctx = canvas.getContext('2d');
      if (!ctx) {
        return;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      try {
        const facePredictions =
          faceTrackingEnabled && faceDetector
            ? await faceDetector.estimateFaces(video, {
                flipHorizontal: false,
                staticImageMode: false,
              })
            : [];

        const handPredictions =
          handTrackingEnabled && handDetector
            ? await handDetector.estimateHands(video, {
                flipHorizontal: false,
                staticImageMode: false,
              })
            : [];
        
        // Debug: verificar estrutura do handedness
        if (handPredictions.length > 0) {
          console.log('Hand predictions:', handPredictions[0]);
        }

        // Inverter coordenadas X para espelhar (vídeo está espelhado, canvas não)
        const flipX = (x: number) => videoWidth - x;

        if (faceTrackingEnabled) {
          drawFaceMesh(
            facePredictions.map((pred: { keypoints: Array<{ x: number; y: number }> }) => ({
              keypoints: pred.keypoints.map((kp: { x: number; y: number }) => [flipX(kp.x), kp.y] as [number, number]),
            })),
            ctx
          );
        }

        if (handTrackingEnabled) {
          // Encontrar mão direita (do ponto de vista do usuário - imagem espelhada)
          const rightHand = handPredictions.find((pred: any) => {
            const handedness = typeof pred.handedness === 'string' 
              ? pred.handedness 
              : pred.handedness?.[0]?.label;
            // Como a imagem está espelhada, Left do modelo = direita do usuário
            return handedness === 'Left';
          });

          // Atualizar posição da mão direita para controlar o cubo
          if (rightHand && rightHand.keypoints && rightHand.keypoints.length > 0) {
            // Usar o ponto central da mão (pulso - índice 0)
            const wrist = rightHand.keypoints[0];
            // Coordenadas originais do modelo (antes de espelhar)
            const originalX = wrist.x;
            const originalY = wrist.y;
            
            // Normalizar coordenadas para 0-1 baseado no tamanho do vídeo
            // Usar coordenadas originais para rotação correta
            const normalizedX = originalX / videoWidth;
            const normalizedY = originalY / videoHeight;
            
            // Para movimento: usar coordenadas espelhadas (flipX)
            // Para rotação: usar coordenadas originais (sem flip)
            const flippedX = flipX(originalX) / videoWidth;
            
            // Converter para coordenadas do cubo dentro do container (0-100%)
            // X: espelhado para movimento, mas rotação usa original
            // Y: invertido para que movimento para cima mova o cubo para cima
            setRightHandPosition({
              x: flippedX * 100, // Posição X espelhada para movimento
              y: (1 - normalizedY) * 100, // Posição Y invertida
              z: 0,
              rotationX: normalizedY * 360, // Rotação X baseada em Y original
              rotationY: normalizedX * 360, // Rotação Y baseada em X original (não espelhado)
            });
          } else {
            setRightHandPosition(null);
          }

          drawHands(
            handPredictions.map((pred: any) => ({
              keypoints: pred.keypoints.map((kp: { x: number; y: number }) => [flipX(kp.x), kp.y] as [number, number]),
              handedness: pred.handedness,
            })),
            ctx
          );
        } else {
          setRightHandPosition(null);
        }
      } catch (err) {
        console.error('Erro na detecção:', err);
      }
    },
    [faceTrackingEnabled, handTrackingEnabled]
  );

  useEffect(() => {
    let faceDetector: FaceLandmarksDetector | null = null;
    let handDetector: HandDetector | null = null;
    let isMounted = true;
    // Capturar referência do webcam no início do effect para usar no cleanup
    const currentWebcam = webcamRef.current;

    const initializeModels = async () => {
      try {
        setLoading({ faceModel: true, handModel: true, camera: false });
        setError(null);
        console.log('Iniciando acesso à câmera...');

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720 },
        });
        console.log('Câmera acessada com sucesso');

        if (webcamRef.current?.video) {
          webcamRef.current.video.srcObject = stream;
          setLoading((prev) => ({ ...prev, camera: true }));
        }

        console.log('Carregando módulos TensorFlow...');
        // Importação dinâmica para garantir que os módulos sejam carregados corretamente
        const [faceModule, handModule] = await Promise.all([
          import('@tensorflow-models/face-landmarks-detection'),
          import('@tensorflow-models/hand-pose-detection'),
        ]);
        console.log('Módulos carregados:', {
          faceModule: !!faceModule,
          handModule: !!handModule,
          faceCreateDetector: typeof faceModule.createDetector,
          handCreateDetector: typeof handModule.createDetector,
        });

        // Verificar se as funções existem
        if (!faceModule.createDetector || typeof faceModule.createDetector !== 'function') {
          throw new Error(`createDetector não encontrado ou não é uma função. Tipo: ${typeof faceModule.createDetector}`);
        }
        if (!handModule.createDetector || typeof handModule.createDetector !== 'function') {
          throw new Error(`createDetector não encontrado ou não é uma função. Tipo: ${typeof handModule.createDetector}`);
        }
        if (!faceModule.SupportedModels) {
          throw new Error('SupportedModels não encontrado em face-landmarks-detection');
        }
        if (!handModule.SupportedModels) {
          throw new Error('SupportedModels não encontrado em hand-pose-detection');
        }

        console.log('Criando detectores...');
        
        // Carregar detectores sequencialmente para evitar conflitos
        // Usar try-catch individual para cada detector
        let faceModel: FaceLandmarksDetector | null = null;
        let handModel: HandDetector | null = null;
        
        try {
          faceModel = await faceModule.createDetector(
            faceModule.SupportedModels.MediaPipeFaceMesh,
            {
              runtime: 'mediapipe' as const,
              solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619',
              refineLandmarks: true,
            }
          );
          console.log('Detector facial criado');
        } catch (faceError: any) {
          // Ignorar erro "File exists" - os arquivos já foram carregados
          if (faceError?.message?.includes('File exists') || faceError?.message?.includes('EEXIST')) {
            console.warn('Arquivos MediaPipe já existem, continuando...');
            // Tentar criar novamente - desta vez deve funcionar
            try {
              faceModel = await faceModule.createDetector(
                faceModule.SupportedModels.MediaPipeFaceMesh,
                {
                  runtime: 'mediapipe' as const,
                  solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@0.4.1633559619',
                  refineLandmarks: true,
                }
              );
              console.log('Detector facial criado (retry)');
            } catch (retryError) {
              console.error('Erro ao criar detector facial após retry:', retryError);
              throw retryError;
            }
          } else {
            console.error('Erro ao criar detector facial:', faceError);
            throw faceError;
          }
        }
        
        try {
          handModel = await handModule.createDetector(
            handModule.SupportedModels.MediaPipeHands,
            {
              runtime: 'mediapipe' as const,
              solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240',
              modelType: 'full' as const,
            }
          );
          console.log('Detector de mãos criado');
        } catch (handError: any) {
          // Ignorar erro "File exists" - os arquivos já foram carregados
          if (handError?.message?.includes('File exists') || handError?.message?.includes('EEXIST')) {
            console.warn('Arquivos MediaPipe já existem, continuando...');
            // Tentar criar novamente - desta vez deve funcionar
            try {
              handModel = await handModule.createDetector(
                handModule.SupportedModels.MediaPipeHands,
                {
                  runtime: 'mediapipe' as const,
                  solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1675469240',
                  modelType: 'full' as const,
                }
              );
              console.log('Detector de mãos criado (retry)');
            } catch (retryError) {
              console.error('Erro ao criar detector de mãos após retry:', retryError);
              throw retryError;
            }
          } else {
            console.error('Erro ao criar detector de mãos:', handError);
            throw handError;
          }
        }
        
        if (!isMounted) {
          return;
        }
        console.log('Detectores criados com sucesso');

        faceDetector = faceModel;
        handDetector = handModel;

        setLoading({ faceModel: false, handModel: false, camera: true });
        console.log('Inicialização completa');

        const runDetection = () => {
          detect(faceDetector, handDetector);
          animationFrameRef.current = requestAnimationFrame(runDetection);
        };

        runDetection();
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : String(err);
        setError(`Erro ao inicializar: ${errorMessage}`);
        console.error('Erro ao acessar câmera ou carregar modelos:', err);
        if (err instanceof Error) {
          console.error('Stack trace:', err.stack);
        }
        setLoading({ faceModel: false, handModel: false, camera: false });
      }
    };

    let isInitialized = false;
    
    initializeModels().then(() => {
      isInitialized = true;
    }).catch(() => {
      // Erro já foi tratado no catch interno
    });

    // Adicionar timeout para evitar travamento indefinido
    const timeoutId = setTimeout(() => {
      if (!isInitialized) {
        console.warn('Timeout ao carregar modelos');
        setError('Timeout ao carregar modelos. Verifique o console para mais detalhes.');
        setLoading({ faceModel: false, handModel: false, camera: false });
      }
    }, 30000); // 30 segundos

    return () => {
      clearTimeout(timeoutId);
    };

    initializeModels();

    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      // Usar a referência capturada no início do effect
      if (currentWebcam?.video?.srcObject) {
        const stream = currentWebcam.video.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [detect]);

  const isLoading = loading.faceModel || loading.handModel || !loading.camera;

  return (
    <div className="app-container">
      {error && (
        <div className="error-message" role="alert">
          {error}
        </div>
      )}
      {isLoading && (
        <div className="loading-overlay">
          <div className="loading-spinner" />
          <p>
            {loading.faceModel && 'Carregando modelo facial...'}
            {!loading.faceModel && loading.handModel && 'Carregando modelo de mãos...'}
            {!loading.faceModel && !loading.handModel && !loading.camera && 'Acessando câmera...'}
          </p>
        </div>
      )}
      <div className="controls-panel">
        <label className="toggle-control">
          <input
            type="checkbox"
            checked={faceTrackingEnabled}
            onChange={(e) => setFaceTrackingEnabled(e.target.checked)}
            disabled={isLoading}
          />
          <span className="toggle-label">Rastreamento Facial</span>
          <span className={`toggle-switch ${faceTrackingEnabled ? 'active' : ''}`} />
        </label>
        <label className="toggle-control">
          <input
            type="checkbox"
            checked={handTrackingEnabled}
            onChange={(e) => setHandTrackingEnabled(e.target.checked)}
            disabled={isLoading}
          />
          <span className="toggle-label">Rastreamento de Mãos</span>
          <span className={`toggle-switch ${handTrackingEnabled ? 'active' : ''}`} />
        </label>
      </div>
      <div className="main-content">
        <div className="video-container">
          <Webcam
            ref={webcamRef}
            className="webcam"
            videoConstraints={{
              width: { ideal: 1280 },
              height: { ideal: 720 },
              facingMode: 'user',
            }}
          />
          <canvas ref={canvasRef} className="canvas-overlay" />
        </div>
        <div className="cube-container">
          <div 
            className="cube-3d"
            style={{
              transform: rightHandPosition
                ? `translate3d(calc(${rightHandPosition.x}% - 50px), calc(${rightHandPosition.y}% - 50px), ${rightHandPosition.z}px) rotateX(${rightHandPosition.rotationX}deg) rotateY(${rightHandPosition.rotationY}deg)`
                : 'translate3d(calc(50% - 50px), calc(50% - 50px), 0px) rotateX(0deg) rotateY(0deg)',
            }}
          >
            <div className="cube-face cube-front"></div>
            <div className="cube-face cube-back"></div>
            <div className="cube-face cube-right"></div>
            <div className="cube-face cube-left"></div>
            <div className="cube-face cube-top"></div>
            <div className="cube-face cube-bottom"></div>
          </div>
          {!rightHandPosition && (
            <div className="cube-placeholder">
              <p>Mostre sua mão direita para controlar o cubo</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;

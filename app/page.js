'use client';

import { useEffect, useRef, useState } from 'react';
import cv from '../services/cv';
import * as faceapi from 'face-api.js';
// import * as tf from "@tensorflow/tfjs";
// import * as handpose from "@tensorflow-models/handpose";
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// We'll limit the processing size to 200px.
const maxVideoSize = 200;
const twMaxVideoSize = `w-[${maxVideoSize}px]`;

const MODEL_URL = '/models';

// const referenceImageSrc = 'knownFaces/obama.jpeg'
// const referenceImageSrc = 'knownFaces/biden.jpeg'
// const referenceImageSrc = 'knownFaces/brumsen.jpeg'
const referenceImageSrc = 'knownFaces/emilio.jpeg';
// const referenceImageSrc = 'knownFaces/emilioNow.png';

const randomNumOfFingers = 2;
let handLandmarker = undefined;
let runningMode = 'IMAGE';

export default function Page() {
  const [isVideoLoaded, setIsVideoLoaded] = useState(false);
  const [isProcessingFaceImage, setIsProcessingFaceImage] = useState(false);
  const [firstFaceDetectionAttempted, setFirstFaceDetectionAttempted] =
    useState(false);
  const [isAnalyzingFaceImage, setIsAnalyzingFaceImage] = useState(false);
  const [isFaceDetected, setIsFaceDetected] = useState(false);
  const [detectedFaceDesc, setDetectedFaceDesc] = useState([]);
  const [faceMatchResult, setFaceMatchResult] = useState(null);
  const [isGestureDetected, setIsGestureDetected] = useState(false);
  const [gestureMatchResult, setGestureMatchResult] = useState(null);
  const videoElement = useRef(null);
  const canvasElFace = useRef(null);
  const canvasElGesture = useRef(null);

  const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm'
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
        delegate: 'GPU',
      },
      runningMode: runningMode,
      numHands: 2,
    });
  };

  const takePhotoAndPrintToCanvas = async (canvas) => {
    const ctx = canvas.current.getContext('2d', {
      willReadFrequently: true,
    });
    ctx.drawImage(videoElement.current, 0, 0, maxVideoSize, maxVideoSize);

    const image = ctx.getImageData(0, 0, maxVideoSize, maxVideoSize);
    // Load the model
    await cv.load();
    // Processing image
    const processedImage = await cv.imageProcessing(image);
    // Render the processed image to the canvas
    ctx.putImageData(processedImage.data.payload, 0, 0);
  };

  const runFaceDetection = async () => {
    await faceapi.loadSsdMobilenetv1Model(MODEL_URL);
    await faceapi.loadFaceLandmarkModel(MODEL_URL);
    await faceapi.loadFaceRecognitionModel(MODEL_URL);
    const input = document.getElementById('facePhotoCanvas');
    const detectedFacesData = await faceapi
      .detectAllFaces(input)
      .withFaceLandmarks()
      .withFaceDescriptors();
    // multiple faces detected
    if (detectedFacesData.length > 1) {
      faceapi.draw.drawDetections(facePhotoCanvas, detectedFacesData);
      faceapi.draw.drawFaceLandmarks(facePhotoCanvas, detectedFacesData);
      // one face detected
    } else if (detectedFacesData.length > 0) {
      setIsFaceDetected(true);
      setDetectedFaceDesc(detectedFacesData[0].descriptor);

      faceapi.draw.drawDetections(facePhotoCanvas, detectedFacesData, {});
      faceapi.draw.drawFaceLandmarks(facePhotoCanvas, detectedFacesData);
    }
  };

  const runFaceRecognition = async () => {
    const facesInMatcher = [];
    const referenceImage = document.getElementById('referenceImage');
    const knownFaceData = await faceapi
      .detectAllFaces(referenceImage)
      .withFaceLandmarks()
      .withFaceDescriptors();
    facesInMatcher.push(knownFaceData);

    // add label and create FaceMatcher with the detection results
    // of the reference image
    const knownLabeledFaceDesc = new faceapi.LabeledFaceDescriptors('emilio', [
      knownFaceData[0].descriptor,
    ]);
    const faceMatcher = new faceapi.FaceMatcher(knownLabeledFaceDesc);

    const matchValue = faceMatcher.findBestMatch(detectedFaceDesc)._distance;
    console.log('matchValue', matchValue);
    if (matchValue < 0.45) {
      setFaceMatchResult('Face Recognized');
      createHandLandmarker();
      // } else if (matchValue < 0.55) {
      // setFaceMatchResult('Please try again');
    } else {
      // setFaceMatchResult('Not even close');
      setFaceMatchResult('Please try again');
    }
  };

  /**
   * What we will do in the onClick event is capture a frame within
   * the video to pass this image on our service.
   */
  const onClickDetectFace = async () => {
    setFaceMatchResult(null);
    setIsFaceDetected(false);
    setIsProcessingFaceImage(true);
    setFirstFaceDetectionAttempted(true);
    await takePhotoAndPrintToCanvas(canvasElFace);
    setIsProcessingFaceImage(false);
    setIsAnalyzingFaceImage(true);
    await runFaceDetection();
    setIsAnalyzingFaceImage(false);
  };

  const onClickRecognizeFace = async () => {
    await runFaceRecognition();
  };

  const onClickDetectGesture = async () => {
    await takePhotoAndPrintToCanvas(canvasElGesture);

    // When an image is clicked, let's detect it and display results!
    if (!handLandmarker) {
      console.log('Wait for handLandmarker to load before clicking!');
      return;
    }

    if (runningMode === 'VIDEO') {
      runningMode = 'IMAGE';
      await handLandmarker.setOptions({ runningMode: 'IMAGE' });
    }
  };

  const onClickRecognizeGesture = async () => {

  };
  /**
   * In the useEffect hook what we are going to do is load the video
   * element so that it plays what you see on the camera. This way
   * it's like a viewer of what the camera sees and then at any
   * time we can capture a frame to take a picture and upload it
   * to OpenCV.
   */
  useEffect(() => {
    async function initCamera() {
      videoElement.current.width = maxVideoSize;
      videoElement.current.height = maxVideoSize;

      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: 'user',
            width: maxVideoSize,
            height: maxVideoSize,
          },
        });
        videoElement.current.srcObject = stream;

        return new Promise((resolve) => {
          videoElement.current.onloadedmetadata = () => {
            resolve(videoElement.current);
          };
        });
      }
      const errorMessage =
        'This browser does not support video capture, or this device does not have a camera';
      alert(errorMessage);
      return Promise.reject(errorMessage);
    }

    async function load() {
      const videoLoaded = await initCamera();
      videoLoaded.play();
      setIsVideoLoaded(true);
      return videoLoaded;
    }

    load();
  }, []);

  /**
   * What we're going to render is:
   *
   * 1. A video component for the user to see what he sees on the camera.
   *
   * 2. A simple button, that with the onClick we will generate an image of
   *  the video, we will load OpenCV and we will treat the image.
   *
   * 3. A canvas, which will allow us to capture the image of the video
   * while showing the user what image has been taken from the video after
   * pressing the button.
   *
   */
  return (
    <div className='flex flex-col items-center justify-center h-screen'>
      <div
        className={`p-2 border-2 border-black rounded w-[200px] text-center bg-red-200 ${
          faceMatchResult === 'Face Recognized' && `bg-green-200`
        }`}
      >
        Face Recognized
      </div>
      <div
        className={`p-2 border-2 border-black rounded w-[200px] text-center bg-red-200 ${
          gestureMatchResult === 'Gesture Recognized' && `bg-green-200`
        }`}
      >
        Gesture Recognized
      </div>
      <br />
      <div
        className={`p-2 border-2 border-black rounded ${twMaxVideoSize} text-center`}
      >
        Show This Face
      </div>
      <img
        id='referenceImage'
        width={maxVideoSize}
        height={maxVideoSize}
        src={referenceImageSrc}
      />

      <canvas
        id='facePhotoCanvas'
        ref={canvasElFace}
        width={maxVideoSize}
        height={maxVideoSize}
      ></canvas>
      {((isVideoLoaded && !isFaceDetected) ||
        (isFaceDetected && faceMatchResult === 'Please try again')) && (
        <button
          className={`p-2 border-2 border-black rounded hover:text-white hover:bg-black ${twMaxVideoSize}`}
          disabled={isProcessingFaceImage || !isVideoLoaded}
          onClick={onClickDetectFace}
        >
          {isProcessingFaceImage || isAnalyzingFaceImage
            ? 'Analyzing...'
            : 'Detect Face'}
        </button>
      )}
      {faceMatchResult === 'Face Recognized' && (
        <>
          <canvas
            id='gesturePhotoCanvas'
            ref={canvasElGesture}
            width={maxVideoSize}
            height={maxVideoSize}
          ></canvas>
          <button
            className={`p-2 border-2 border-black rounded hover:text-white hover:bg-black ${twMaxVideoSize}`}
            onClick={onClickDetectGesture}
          >
            Detect Gesture
          </button>
          {isGestureDetected && (
            <>
              <button
                className={`p-2 border-2 border-black rounded hover:text-white hover:bg-black ${twMaxVideoSize}`}
                onClick={onClickRecognizeGesture}
              >
                {gestureMatchResult ?? 'Recognize Gesture'}
              </button>
            </>
          )}
        </>
      )}
      <video className='video' playsInline ref={videoElement} />
      {!isFaceDetected &&
        firstFaceDetectionAttempted &&
        !isProcessingFaceImage &&
        !isAnalyzingFaceImage && (
          <div
            className={`p-2 border-2 border-black rounded ${twMaxVideoSize} text-center`}
          >
            No Face Detected
          </div>
        )}
      {isFaceDetected && !faceMatchResult && (
        <button
          className={`p-2 border-2 border-black rounded hover:text-white hover:bg-black ${twMaxVideoSize}`}
          onClick={onClickRecognizeFace}
        >
          Recognize Face
        </button>
      )}
      {faceMatchResult === 'Please try again' && (
        <div
          className={`p-2 border-2 border-black rounded ${twMaxVideoSize} text-center`}
        >
          Please try again
        </div>
      )}
      {faceMatchResult === 'Face Recognized' &&
        gestureMatchResult !==
          'Gesture Recognized'(
            <div
              className={`p-2 border-2 border-black rounded ${twMaxVideoSize} text-center`}
            >
              Hold Up {randomNumOfFingers} Fingers
            </div>
          )}
      {isGestureDetected && !gestureMatchResult && (
        <button
          className={`p-2 border-2 border-black rounded hover:text-white hover:bg-black ${twMaxVideoSize}`}
          onClick={onClickRecognizeGesture}
        >
          Recognize Gesture
        </button>
      )}
    </div>
  );
}

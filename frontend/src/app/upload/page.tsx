'use client';

import React, { useState, useRef } from 'react';
import Image from 'next/image';
import {
  UploadCloud,
  CheckCircle2,
  AlertTriangle,
  Sparkles,
  Zap,
  Eye,
  Cpu,
  Layers,
  ArrowRight,
  Info,
  RefreshCw,
  XCircle,
} from 'lucide-react';
import { predictBreed, PredictResponse } from '@/lib/api';

export default function UploadPage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [activeTab, setActiveTab] = useState<'overlay' | 'original' | 'heatmap'>('overlay');
  const [modelVersion, setModelVersion] = useState<string>('selected_breeds_swin_v2');

  const handleFileSelect = (file: File) => {
    setError(null);
    setResult(null);

    // Validate File Size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      setError(`File size (${(file.size / (1024 * 1024)).toFixed(2)}MB) exceeds 10MB limit.`);
      return;
    }

    // Validate File Type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff'];
    if (!validTypes.includes(file.type.toLowerCase())) {
      setError(`Unsupported image format '${file.type}'. Allowed formats: JPEG, PNG, WebP, BMP, TIFF.`);
      return;
    }

    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    try {
      const res = await predictBreed(selectedFile, true, modelVersion);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'AI prediction pipeline failed. Please check backend connection.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center max-w-2xl mx-auto space-y-2">
        <h1 className="text-3xl font-extrabold text-white tracking-tight">
          Livestock Image Upload & AI Analysis
        </h1>
        <p className="text-xs text-slate-400">
          Upload an image of an Indian cattle or buffalo to run YOLO animal detection, EfficientNet-B0 breed classification, and Grad-CAM explainability.
        </p>
      </div>

      {/* Error Alert Banner */}
      {error && (
        <div className="max-w-3xl mx-auto p-4 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center space-x-3 shadow-lg">
          <XCircle className="w-5 h-5 text-rose-400 flex-shrink-0" />
          <div className="flex-1">{error}</div>
        </div>
      )}

      {/* Upload Zone Section (Shown when no result exists) */}
      {!result && (
        <div className="max-w-2xl mx-auto space-y-6">
          <div className="flex items-center justify-between p-3.5 rounded-2xl bg-slate-900/80 border border-slate-800">
            <div className="flex items-center space-x-2">
              <Cpu className="w-4 h-4 text-indigo-400" />
              <span className="text-xs font-semibold text-white">Active Classification Model:</span>
            </div>
            <select
              value={modelVersion}
              onChange={(e) => setModelVersion(e.target.value)}
              className="bg-slate-950 border border-slate-700 text-xs text-indigo-300 font-medium rounded-lg px-3 py-1.5 focus:outline-none focus:border-indigo-500"
            >
              <option value="selected_breeds_swin_v2">Swin-T V2 [Production Default] (8 Selected Breeds - 78.79% Top-1 / 90.91% Top-3)</option>
              <option value="selected_breeds_swin_synth_v3">Swin-T Synth V3 [Experimental] (8 Selected Breeds - 75.76% Top-1 / 100.00% Top-3)</option>
              <option value="convnext_v2">ConvNeXt-Tiny (8 Selected Breeds - 72.73% Top-1 / 93.94% Top-3)</option>
              <option value="selected_breeds_v4">EfficientNet-B4 (8 Selected Breeds - 51.52% Top-1 / 75.76% Top-3)</option>
              <option value="selected_breeds_v2">EfficientNet-B0 V2 (8 Selected Breeds - 69.70% Top-1 / 96.97% Top-3)</option>
              <option value="high_accuracy_v2">High-Accuracy V2 (82 Breeds Exploratory)</option>
            </select>
          </div>

          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="glass-card p-10 rounded-3xl border-2 border-dashed border-slate-700 hover:border-indigo-500/80 text-center cursor-pointer transition group relative"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
            />

            {previewUrl ? (
              <div className="space-y-4">
                <img
                  src={previewUrl}
                  alt="Selected Preview"
                  className="max-h-64 mx-auto rounded-2xl object-contain shadow-xl border border-slate-700"
                />
                <p className="text-xs font-semibold text-indigo-300">{selectedFile?.name}</p>
                <p className="text-[11px] text-slate-400">
                  {((selectedFile?.size || 0) / 1024).toFixed(1)} KB • Click or drop to replace
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="w-16 h-16 rounded-2xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center mx-auto group-hover:scale-110 transition">
                  <UploadCloud className="w-8 h-8 text-indigo-400" />
                </div>
                <div>
                  <p className="text-sm font-bold text-white">Drag & drop livestock image here</p>
                  <p className="text-xs text-slate-400 mt-1">or click to browse files from your computer</p>
                </div>
                <p className="text-[11px] text-slate-500">
                  Supports JPEG, PNG, WebP, BMP, TIFF (Max size: 10MB)
                </p>
              </div>
            )}
          </div>

          {selectedFile && (
            <div className="flex items-center justify-center space-x-4">
              <button
                onClick={handleAnalyze}
                disabled={loading}
                className="px-8 py-3.5 rounded-xl bg-gradient-to-r from-indigo-600 via-purple-600 to-emerald-500 text-white font-semibold text-sm shadow-xl shadow-indigo-600/30 hover:scale-105 transition flex items-center space-x-2 disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    <span>Executing AI Pipeline...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    <span>Analyze Breed with AI</span>
                  </>
                )}
              </button>

              <button
                onClick={handleReset}
                disabled={loading}
                className="px-5 py-3 rounded-xl bg-slate-900 border border-slate-800 text-slate-400 hover:text-white text-xs font-semibold hover:bg-slate-800 transition"
              >
                Reset
              </button>
            </div>
          )}
        </div>
      )}

      {/* Loading Skeleton View */}
      {loading && (
        <div className="max-w-4xl mx-auto glass-card p-8 rounded-3xl space-y-6 text-center animate-pulse">
          <div className="w-12 h-12 rounded-2xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center mx-auto">
            <Cpu className="w-6 h-6 text-indigo-400 animate-spin" />
          </div>
          <h2 className="text-xl font-bold text-white">Processing Image through AI Models...</h2>
          <div className="max-w-md mx-auto space-y-2 text-xs text-slate-400">
            <p>Stage 1: Running YOLOv8 Animal Detection</p>
            <p>Stage 2: OpenCV Letterbox Crop & Preprocessing</p>
            <p>Stage 3: EfficientNet-B0 Breed Classification</p>
            <p>Stage 4: Computing Grad-CAM Attention Heatmap</p>
          </div>
        </div>
      )}

      {/* Prediction Result View */}
      {result && !loading && (
        <div className="max-w-5xl mx-auto space-y-8">
          {/* Status Banners */}
          {result.prediction_status === 'no_animal_detected' && (
            <div className="p-4 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-center space-x-3">
              <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0" />
              <div>
                <p className="font-bold">No Animal Bounding Box Detected by YOLO</p>
                <p className="text-[11px] text-amber-400/80">
                  Fallback full-image crop was analyzed by EfficientNet-B0. Ensure the animal is clearly visible in the frame.
                </p>
              </div>
            </div>
          )}

          {result.prediction_status === 'multiple_animals_detected' && (
            <div className="p-4 rounded-2xl bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 text-xs flex items-center space-x-3">
              <Info className="w-5 h-5 text-indigo-400 flex-shrink-0" />
              <div>
                <p className="font-bold">Multiple Livestock Animals Detected in Image</p>
                <p className="text-[11px] text-indigo-300/80">
                  Primary animal detection ROI was prioritized for breed classification.
                </p>
              </div>
            </div>
          )}

          {/* Top Result Card Header */}
          <div className="glass-card p-8 rounded-3xl space-y-6">
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 pb-6 border-b border-slate-800">
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="px-2.5 py-1 rounded-md bg-indigo-500/10 text-indigo-300 text-xs font-semibold uppercase border border-indigo-500/20">
                    {result.animal_type}
                  </span>
                  <span className="px-2.5 py-1 rounded-md bg-emerald-500/10 text-emerald-300 text-[11px] font-semibold border border-emerald-500/20">
                    {result.model_version || modelVersion}
                  </span>
                  <span className="text-xs text-slate-400 font-medium">
                    YOLO: {(result.animal_confidence * 100).toFixed(1)}% • Latency: {result.inference_time.total_ms.toFixed(1)}ms
                  </span>
                </div>
                <h2 className="text-3xl font-extrabold text-white mt-2">
                  Predicted Breed: <span className="gradient-text">{result.predicted_breed}</span>
                </h2>
              </div>

              <div className="text-right">
                <div className="text-3xl font-black text-emerald-400">
                  {(result.breed_confidence * 100).toFixed(1)}%
                </div>
                <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mt-0.5">
                  Top-1 Confidence Score
                </div>
              </div>
            </div>

            {/* Visual Explainability & Images Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {/* Left Column: Grad-CAM Viewer */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                    <Sparkles className="w-4 h-4 text-purple-400" />
                    <span>Grad-CAM Explainability Heatmap</span>
                  </h3>

                  {/* Tabs */}
                  {result.gradcam_output && (
                    <div className="flex items-center space-x-1 bg-slate-900 p-1 rounded-lg border border-slate-800 text-[11px]">
                      <button
                        onClick={() => setActiveTab('overlay')}
                        className={`px-2.5 py-1 rounded-md font-semibold transition ${
                          activeTab === 'overlay'
                            ? 'bg-indigo-600 text-white'
                            : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        Overlay
                      </button>
                      <button
                        onClick={() => setActiveTab('heatmap')}
                        className={`px-2.5 py-1 rounded-md font-semibold transition ${
                          activeTab === 'heatmap'
                            ? 'bg-indigo-600 text-white'
                            : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        Heatmap
                      </button>
                      <button
                        onClick={() => setActiveTab('original')}
                        className={`px-2.5 py-1 rounded-md font-semibold transition ${
                          activeTab === 'original'
                            ? 'bg-indigo-600 text-white'
                            : 'text-slate-400 hover:text-white'
                        }`}
                      >
                        Original
                      </button>
                    </div>
                  )}
                </div>

                <div className="glass-card p-2 rounded-2xl border border-slate-800 text-center relative overflow-hidden min-h-[280px] flex items-center justify-center">
                  {activeTab === 'overlay' && result.gradcam_output?.overlay_base64 && (
                    <img
                      src={result.gradcam_output.overlay_base64}
                      alt="Grad-CAM Overlay"
                      className="max-h-72 rounded-xl object-contain mx-auto shadow-lg"
                    />
                  )}
                  {activeTab === 'heatmap' && result.gradcam_output?.heatmap_base64 && (
                    <img
                      src={result.gradcam_output.heatmap_base64}
                      alt="Grad-CAM Heatmap"
                      className="max-h-72 rounded-xl object-contain mx-auto shadow-lg"
                    />
                  )}
                  {(activeTab === 'original' || !result.gradcam_output) && previewUrl && (
                    <img
                      src={previewUrl}
                      alt="Original Input"
                      className="max-h-72 rounded-xl object-contain mx-auto shadow-lg"
                    />
                  )}
                </div>
                <p className="text-[11px] text-slate-400 italic text-center">
                  Red/Yellow areas highlight facial structure and body features driving EfficientNet breed decision.
                </p>
              </div>

              {/* Right Column: Top-3 Rankings & Latency */}
              <div className="space-y-6">
                <div className="space-y-3">
                  <h3 className="text-sm font-bold text-white">Top-3 Ranked Breed Candidates</h3>
                  <div className="space-y-3">
                    {result.top_3_predictions.map((p, idx) => (
                      <div key={idx} className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1.5">
                        <div className="flex items-center justify-between text-xs font-semibold">
                          <span className="text-white">
                            {idx + 1}. {p.display_name}
                          </span>
                          <span className="text-indigo-400">{(p.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-indigo-500 to-emerald-400 rounded-full transition-all duration-500"
                            style={{ width: `${Math.max(p.confidence * 100, 4)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Latency Breakdown */}
                <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800 space-y-2 text-xs">
                  <div className="font-bold text-white flex items-center justify-between">
                    <span>Inference Latency Breakdown</span>
                    <span className="text-emerald-400">{result.inference_time.total_ms.toFixed(1)} ms Total</span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-[11px] text-slate-400 pt-2 border-t border-slate-800">
                    <div>
                      <span className="block text-slate-500">Detection</span>
                      <span className="font-semibold text-slate-200">{result.inference_time.detection_ms.toFixed(1)} ms</span>
                    </div>
                    <div>
                      <span className="block text-slate-500">Classifier</span>
                      <span className="font-semibold text-slate-200">{result.inference_time.classification_ms.toFixed(1)} ms</span>
                    </div>
                    <div>
                      <span className="block text-slate-500">Grad-CAM</span>
                      <span className="font-semibold text-slate-200">{result.inference_time.gradcam_ms.toFixed(1)} ms</span>
                    </div>
                  </div>
                </div>

                <div className="pt-2 flex justify-center">
                  <button
                    onClick={handleReset}
                    className="px-6 py-2.5 rounded-xl bg-indigo-600 text-white font-semibold text-xs shadow-lg hover:bg-indigo-500 transition flex items-center space-x-2"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    <span>Analyze Another Image</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

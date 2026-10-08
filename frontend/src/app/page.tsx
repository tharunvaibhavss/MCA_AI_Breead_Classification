import React from 'react';
import Link from 'next/link';
import {
  UploadCloud,
  Sparkles,
  ShieldCheck,
  Eye,
  Layers,
  Cpu,
  ArrowRight,
  CheckCircle2,
} from 'lucide-react';

export default function HomePage() {
  const breedsList = [
    { name: 'Dangi', type: 'Cattle', origin: 'Maharashtra / Gujarat', badge: 'Draught' },
    { name: 'Gaolao', type: 'Cattle', origin: 'Maharashtra / MP', badge: 'Dual-Purpose' },
    { name: 'Gir', type: 'Cattle', origin: 'Gujarat', badge: 'Dairy' },
    { name: 'Ongole', type: 'Cattle', origin: 'Andhra Pradesh', badge: 'Dual-Purpose' },
    { name: 'Sahiwal', type: 'Cattle', origin: 'Punjab', badge: 'Dairy' },
    { name: 'Manda', type: 'Buffalo', origin: 'Odisha / AP', badge: 'Hardy Draught' },
    { name: 'Surti', type: 'Buffalo', origin: 'Gujarat', badge: 'Compact Dairy' },
    { name: 'Toda', type: 'Buffalo', origin: 'Tamil Nadu (Nilgiris)', badge: 'Pastoral Heritage' },
  ];

  const pipelineSteps = [
    {
      step: '01',
      title: 'YOLO Animal Detection',
      desc: 'Localizes cattle and buffalo in the uploaded image, returning high-confidence bounding box coordinates.',
      icon: Eye,
      tag: 'YOLOv8',
    },
    {
      step: '02',
      title: 'OpenCV Aspect Preprocessing',
      desc: 'Scales and letterboxes cropped animal ROI to 224x224 RGB tensors without aspect ratio distortion.',
      icon: Layers,
      tag: 'OpenCV 4',
    },
    {
      step: '03',
      title: 'Swin Transformer Classification',
      desc: 'Swin-T backbone extracts hierarchical vision transformer features with 78.79% Top-1 & 100% Top-3 capability.',
      icon: Cpu,
      tag: 'Swin-T V2',
    },
    {
      step: '04',
      title: 'Grad-CAM Explainability',
      desc: 'Generates visual attention heatmaps overlaying key anatomical features (dewlap, horns, hump) justifying model predictions.',
      icon: Sparkles,
      tag: 'Grad-CAM',
    },
  ];

  return (
    <div className="space-y-16">
      {/* Hero Section */}
      <section className="relative overflow-hidden pt-8 pb-12 rounded-3xl glass-card border border-slate-800/80 px-6 sm:px-12 text-center">
        <div className="absolute inset-0 bg-gradient-to-tr from-indigo-600/10 via-purple-600/10 to-emerald-500/10 pointer-events-none" />

        <div className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-xs font-semibold mb-6">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Swin Transformer V2 • 78.79% Top-1 / 100% Top-3</span>
        </div>

        <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight max-w-4xl mx-auto leading-tight">
          Intelligent Breed Recognition for{' '}
          <span className="gradient-text">Indian Cattle & Buffaloes</span>
        </h1>

        <p className="mt-6 text-slate-300 text-lg sm:text-xl max-w-2xl mx-auto font-normal leading-relaxed">
          Upload any livestock photo to automatically detect animals, classify indigenous breeds with confidence scoring, and inspect visual Grad-CAM explanations.
        </p>

        <div className="mt-8 flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link
            href="/upload"
            className="w-full sm:w-auto px-8 py-3.5 rounded-xl bg-gradient-to-r from-indigo-600 via-purple-600 to-indigo-700 text-white font-semibold shadow-lg shadow-indigo-600/30 hover:scale-105 transition-all flex items-center justify-center space-x-2"
          >
            <UploadCloud className="w-5 h-5" />
            <span>Upload Image & Scan Breed</span>
          </Link>
          <Link
            href="/breeds"
            className="w-full sm:w-auto px-8 py-3.5 rounded-xl bg-slate-900 border border-slate-700 text-slate-200 font-semibold hover:bg-slate-800 hover:text-white transition flex items-center justify-center space-x-2"
          >
            <span>Explore 8 Selected Breeds</span>
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>

        {/* Feature Badges */}
        <div className="mt-12 pt-8 border-t border-slate-800/80 grid grid-cols-2 md:grid-cols-4 gap-4 text-xs text-slate-400">
          <div className="flex items-center justify-center space-x-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            <span>8 Selected Breeds</span>
          </div>
          <div className="flex items-center justify-center space-x-2">
            <CheckCircle2 className="w-4 h-4 text-indigo-400" />
            <span>Top-3 Probability Scoring</span>
          </div>
          <div className="flex items-center justify-center space-x-2">
            <CheckCircle2 className="w-4 h-4 text-purple-400" />
            <span>Grad-CAM Visual Heatmaps</span>
          </div>
          <div className="flex items-center justify-center space-x-2">
            <CheckCircle2 className="w-4 h-4 text-amber-400" />
            <span>Edge Case Robustness</span>
          </div>
        </div>
      </section>

      {/* AI Pipeline Architecture Grid */}
      <section className="space-y-8">
        <div className="text-center space-y-2">
          <h2 className="text-3xl font-bold tracking-tight text-white">
            End-to-End Deep Learning Architecture
          </h2>
          <p className="text-slate-400 text-sm max-w-xl mx-auto">
            A 4-stage pipeline combining object detection, vision preprocessing, transfer learning, and model explainability.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {pipelineSteps.map((step) => {
            const Icon = step.icon;
            return (
              <div
                key={step.step}
                className="glass-card glass-card-hover p-6 rounded-2xl space-y-4 relative"
              >
                <div className="flex items-center justify-between">
                  <span className="text-2xl font-black text-indigo-500/40">{step.step}</span>
                  <span className="px-2.5 py-1 rounded-md bg-indigo-500/10 text-indigo-300 text-[10px] font-semibold uppercase border border-indigo-500/20">
                    {step.tag}
                  </span>
                </div>
                <div className="w-12 h-12 rounded-xl bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center">
                  <Icon className="w-6 h-6 text-indigo-400" />
                </div>
                <h3 className="text-lg font-bold text-white">{step.title}</h3>
                <p className="text-slate-400 text-xs leading-relaxed">{step.desc}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* Supported Breeds Preview */}
      <section className="space-y-8">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-3xl font-bold tracking-tight text-white">Supported Indigenous Breeds</h2>
            <p className="text-slate-400 text-sm mt-1">
              Optimized for top Indian cattle and buffalo breeds.
            </p>
          </div>
          <Link
            href="/breeds"
            className="text-xs font-semibold text-indigo-400 hover:text-indigo-300 flex items-center space-x-1"
          >
            <span>View Full Characteristics</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {breedsList.map((breed) => (
            <div
              key={breed.name}
              className="glass-card glass-card-hover p-6 rounded-2xl space-y-3 flex flex-col justify-between"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold px-2.5 py-1 rounded-md bg-slate-800 text-slate-300 border border-slate-700">
                  {breed.type}
                </span>
                <span className="text-[11px] font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full border border-emerald-500/20">
                  {breed.badge}
                </span>
              </div>
              <div>
                <h3 className="text-xl font-bold text-white">{breed.name}</h3>
                <p className="text-xs text-slate-400 mt-1">Native State: {breed.origin}</p>
              </div>
              <Link
                href="/breeds"
                className="text-xs font-medium text-indigo-400 hover:text-indigo-300 flex items-center space-x-1 pt-2 border-t border-slate-800"
              >
                <span>View Milk Yield & Features</span>
                <ArrowRight className="w-3 h-3" />
              </Link>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

"use client";

import Link from "next/link";
import { useState } from "react";
import ColorScale from "~/components/ColorScale";
import FeatureMap from "~/components/FeatureMap";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Progress } from "~/components/ui/progress";
import Waveform from "~/components/Waveform";

// --- Interfaces ---
interface Predictions {
  class: string;
  confidence: number;
}

interface LayerData {
  shape: number[];
  values: number[][];
}

interface VisualizationData {
  [layerName: string]: LayerData;
}

interface WaveformData {
  values: number[];
  sample_rate: number;
  duration: number;
}

interface ApiResponse {
  predictions: Predictions[];
  visualizations: VisualizationData; 
  input_spectrogram: LayerData;
  waveform: WaveformData;
}

// --- Constants ---
const ESC50_EMOJI_MAP: Record<string, string> = {
  // Animals
  "dog": "🐕", "rooster": "🐓", "pig": "🐷", "cow": "🐄", "frog": "🐸",
  "cat": "🐱", "hen": "🐔", "insects": "🦗", "sheep": "🐑", "crow": "🦅",
  // Nature
  "rain": "🌧️", "sea_waves": "🌊", "crackling_fire": "🔥", "crickets": "🦗",
  "chirping_birds": "🐦", "water_drops": "💧", "wind": "💨", "pouring_water": "🚰",
  "toilet_flush": "🚽", "thunderstorm": "⛈️",
  // Human
  "crying_baby": "👶", "sneezing": "🤧", "clapping": "👏", "breathing": "😮‍💨",
  "coughing": "🤒", "footsteps": "👣", "laughing": "😂", "brushing_teeth": "🪥",
  "snoring": "😴", "drinking_sipping": "🥤",
  // Interior
  "door_wood_knock": "🚪", "mouse_click": "🖱️", "keyboard_typing": "⌨️",
  "door_wood_creaks": "🚪", "can_opening": "🥫", "washing_machine": "🧺",
  "vacuum_cleaner": "🧹", "clock_alarm": "⏰", "clock_tick": "⏱️", "glass_breaking": "💥",
  // Exterior
  "helicopter": "🚁", "chainsaw": "🪚", "siren": "🚨", "car_horn": "🚗",
  "engine": "🏎️", "train": "🚂", "church_bells": "🔔", "airplane": "✈️",
  "fireworks": "🎆", "hand_saw": "🪚",
};

function getEmojiForClass(className: string): string {
  return ESC50_EMOJI_MAP[className] || "🎵";
}

// --- Logic to Split Main vs Internal Layers ---
function splitLayers(visualizations: VisualizationData | undefined) {
  const main: [string, LayerData][] = [];
  const internals: Record<string, [string, LayerData][]> = {};

  if (!visualizations) {
    return { main, internals };
  }

  const keys = Object.keys(visualizations).sort();

  for (const name of keys) {
    const data = visualizations[name];
    
    if (!data) continue;

    if (!name.includes(".")) {
      main.push([name, data]);
    } else {
      const parts = name.split(".");
      const parent = parts[0];
      
      if (parent) {
        if (!internals[parent]) {
          internals[parent] = [];
        }
        internals[parent].push([name, data]);
      }
    }
  }
  return { main, internals };
}

export default function HomePage() {
  const [vizData, setVizData] = useState<ApiResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setIsLoading(true);
    setError(null);
    setVizData(null);

    const reader = new FileReader();
    reader.readAsArrayBuffer(file);

    reader.onload = async () => {
      try {
        const arrayBuffer = reader.result as ArrayBuffer;
        const base64String = btoa(
          new Uint8Array(arrayBuffer).reduce(
            (data, byte) => data + String.fromCharCode(byte),
            "",
          ),
        );

        const response = await fetch(
          "https://aranjan45--audio-cnn-inference-audioclassifier-inference.modal.run/",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ audio_data: base64String }),
          }
        );

        if (!response.ok) {
          throw new Error(`API error: ${response.statusText}`);
        }

        const data: ApiResponse = await response.json();
        setVizData(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "An unknown error occurred");
      } finally {
        setIsLoading(false);
      }
    };

    reader.onerror = () => {
      setError("Failed to read file");
      setIsLoading(false);
    };
  };

  const { main, internals } = splitLayers(vizData?.visualizations);

  const allKeys = vizData?.visualizations ? Object.keys(vizData.visualizations) : [];

  return (
    <div className="min-h-screen bg-linear-to-br from-slate-50 via-blue-50 to-indigo-50 relative overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-72 h-72 bg-blue-300/20 rounded-full blur-3xl animate-pulse" style={{ animationDuration: '4s' }}></div>
        <div className="absolute bottom-20 right-10 w-96 h-96 bg-indigo-300/20 rounded-full blur-3xl animate-pulse" style={{ animationDuration: '6s', animationDelay: '1s' }}></div>
        <div className="absolute top-1/2 left-1/2 w-64 h-64 bg-purple-300/10 rounded-full blur-3xl animate-pulse" style={{ animationDuration: '5s', animationDelay: '2s' }}></div>
      </div>

      <div className="container mx-auto p-4 min-h-screen pb-20 relative z-10">
        <div className="mx-auto max-w-6xl space-y-8">
          
          {/* Header */}
          <div className="text-center py-12 space-y-4 animate-in fade-in slide-in-from-top-4 duration-700">
            <div className="inline-flex items-center gap-3 mb-2">
              <div className="w-12 h-12 rounded-2xl bg-linear-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg">
                <svg className="w-7 h-7 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
                </svg>
              </div>
              <h1 className="text-5xl font-black tracking-tight bg-linear-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-transparent">
                Audify
              </h1>
            </div>
            <p className="text-lg text-slate-600 max-w-2xl mx-auto font-medium">
              Peer inside the neural network's decision-making process. Upload audio and watch how convolutional layers transform sound into predictions.
            </p>
          </div>

          {/* Upload Section */}
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-700" style={{ animationDelay: '100ms' }}>
            <Card className="border-2 border-dashed border-blue-300 bg-white/80 backdrop-blur-sm shadow-xl hover:shadow-2xl transition-all duration-300 hover:border-blue-400">
              <CardContent className="pt-8 pb-8">
                <div className="flex flex-col items-center justify-center space-y-6">
                  <div className="relative">
                    <input
                      type="file"
                      accept=".wav"
                      onChange={handleFileChange}
                      className="hidden"
                      id="audio-upload"
                      disabled={isLoading}
                    />
                    <label htmlFor="audio-upload">
                      <Button 
                        asChild 
                        disabled={isLoading} 
                        size="lg"
                        className="cursor-pointer bg-linear-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-lg hover:shadow-xl transition-all duration-300 px-8 py-6 text-base font-semibold"
                      >
                        <span className="flex items-center gap-3">
                          {isLoading ? (
                            <>
                              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                              </svg>
                              Analyzing Audio...
                            </>
                          ) : (
                            <>
                              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                              </svg>
                              Select Audio File
                            </>
                          )}
                        </span>
                      </Button>
                    </label>
                  </div>
                  
                  {fileName && (
                    <div className="flex items-center gap-2 px-4 py-2 bg-blue-50 rounded-lg border border-blue-200 animate-in fade-in slide-in-from-bottom-2">
                      <svg className="w-4 h-4 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
                      </svg>
                      <span className="text-sm font-medium text-blue-900">{fileName}</span>
                    </div>
                  )}
                  
                  {error && (
                    <div className="flex items-center gap-2 px-4 py-3 bg-red-50 rounded-lg border border-red-200 text-red-800 animate-in fade-in slide-in-from-bottom-2">
                      <svg className="w-5 h-5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                      </svg>
                      <span className="text-sm font-medium">{error}</span>
                    </div>
                  )}
                  
                  <p className="text-xs text-slate-500 flex items-center gap-2">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Supported format: WAV files only
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Results Section */}
          {vizData && (
            <div className="space-y-8">
              
              {/* Top Predictions */}
              {vizData.predictions && (
                <div className="animate-in fade-in slide-in-from-bottom-4 duration-700" style={{ animationDelay: '200ms' }}>
                  <Card className="border-0 shadow-2xl bg-white/90 backdrop-blur-sm overflow-hidden">
                    <div className="bg-linear-to-r from-blue-600 to-indigo-600 h-1"></div>
                    <CardHeader className="pb-4">
                      <CardTitle className="text-2xl font-bold flex items-center gap-2">
                        <div className="w-8 h-8 rounded-lg bg-linear-to-br from-blue-500 to-indigo-600 flex items-center justify-center">
                          <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
                            <path d="M2 11a1 1 0 011-1h2a1 1 0 011 1v5a1 1 0 01-1 1H3a1 1 0 01-1-1v-5zM8 7a1 1 0 011-1h2a1 1 0 011 1v9a1 1 0 01-1 1H9a1 1 0 01-1-1V7zM14 4a1 1 0 011-1h2a1 1 0 011 1v12a1 1 0 01-1 1h-2a1 1 0 01-1-1V4z" />
                          </svg>
                        </div>
                        Classification Results
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-5">
                      {vizData.predictions.slice(0, 3).map((pred, index) => (
                        <div 
                          key={pred.class} 
                          className="space-y-3 p-4 rounded-xl bg-linear-to-r from-slate-50 to-blue-50 border border-blue-100 hover:shadow-md transition-all duration-300 animate-in fade-in slide-in-from-left-4"
                          style={{ animationDelay: `${300 + index * 100}ms` }}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-4">
                              <div className="text-4xl bg-white p-3 rounded-xl shadow-sm border border-slate-200">
                                {getEmojiForClass(pred.class)}
                              </div>
                              <div>
                                <span className="font-bold text-lg text-slate-900 capitalize block">
                                  {pred.class.replaceAll("_", " ")}
                                </span>
                                <span className="text-xs text-slate-500">
                                  Rank #{index + 1}
                                </span>
                              </div>
                            </div>
                            <Badge 
                              variant={pred.confidence > 0.8 ? "default" : "secondary"}
                              className={`text-base px-4 py-2 font-bold ${
                                pred.confidence > 0.8 
                                  ? 'bg-linear-to-r from-green-500 to-emerald-600 text-white' 
                                  : 'bg-slate-200 text-slate-700'
                              }`}
                            >
                              {(pred.confidence * 100).toFixed(1)}%
                            </Badge>
                          </div>
                          <div className="relative">
                            <Progress 
                              value={pred.confidence * 100} 
                              className="h-3 bg-slate-200"
                            />
                            <div 
                              className="absolute top-0 left-0 h-3 bg-linear-to-r from-blue-500 via-indigo-500 to-purple-500 rounded-full transition-all duration-1000 ease-out"
                              style={{ width: `${pred.confidence * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                </div>
              )}

              {/* Input Spectrogram & Waveform */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-in fade-in slide-in-from-bottom-4 duration-700" style={{ animationDelay: '400ms' }}>
                {vizData.input_spectrogram && (
                  <Card className="border-0 shadow-xl bg-white/90 backdrop-blur-sm overflow-hidden hover:shadow-2xl transition-shadow duration-300">
                    <div className="bg-linear-to-r from-purple-500 to-pink-500 h-1"></div>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-xl font-bold flex items-center gap-2">
                        <div className="w-7 h-7 rounded-lg bg-linear-to-br from-purple-500 to-pink-600 flex items-center justify-center">
                          <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
                          </svg>
                        </div>
                        Input Spectrogram
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="flex flex-col items-center space-y-4">
                      <div className="w-full p-4 rounded-lg bg-linear-to-br from-slate-50 to-slate-100 border border-slate-200">
                        <FeatureMap 
                          data={vizData.input_spectrogram.values} 
                          title={`Shape: ${vizData.input_spectrogram.shape?.join(" × ")}`}
                           spectrogram={true}
                        />
                      </div>
                      <div className="w-full flex justify-center">
                        <ColorScale width={200} height={12} min={-1} max={1}/>
                      </div>
                    </CardContent>
                  </Card>
                )}
                
                <Card className="border-0 shadow-xl bg-white/90 backdrop-blur-sm overflow-hidden hover:shadow-2xl transition-shadow duration-300">
                  <div className="bg-linear-to-r from-cyan-500 to-blue-500 h-1"></div>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-xl font-bold flex items-center gap-2">
                      <div className="w-7 h-7 rounded-lg bg-linear-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
                        <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M9.383 3.076A1 1 0 0110 4v12a1 1 0 01-1.707.707L4.586 13H2a1 1 0 01-1-1V8a1 1 0 011-1h2.586l3.707-3.707a1 1 0 011.09-.217zM14.657 2.929a1 1 0 011.414 0A9.972 9.972 0 0119 10a9.972 9.972 0 01-2.929 7.071 1 1 0 01-1.414-1.414A7.971 7.971 0 0017 10c0-2.21-.894-4.208-2.343-5.657a1 1 0 010-1.414zm-2.829 2.828a1 1 0 011.415 0A5.983 5.983 0 0115 10a5.984 5.984 0 01-1.757 4.243 1 1 0 01-1.415-1.415A3.984 3.984 0 0013 10a3.983 3.983 0 00-1.172-2.828 1 1 0 010-1.415z" clipRule="evenodd" />
                        </svg>
                      </div>
                      Audio Waveform
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="flex flex-col items-center justify-center h-full min-h-62.5">
                    <Waveform data={vizData.waveform.values} title={`${vizData.waveform.duration.toFixed(2)}s * ${vizData.waveform.sample_rate} Hz`} />
                      
                  </CardContent>
                </Card>
              </div>

              {/* Convolutional Layer Outputs - Hierarchical View */}
              <div className="animate-in fade-in slide-in-from-bottom-4 duration-700" style={{ animationDelay: '600ms' }}>
                <Card className="border-0 shadow-2xl bg-white/90 backdrop-blur-sm overflow-hidden">
                  <div className="bg-linear-to-r from-orange-500 via-red-500 to-pink-500 h-1"></div>
                  <CardHeader className="pb-4">
                    <CardTitle className="text-2xl font-bold flex items-center gap-2">
                      <div className="w-8 h-8 rounded-lg bg-linear-to-br from-orange-500 to-red-600 flex items-center justify-center">
                        <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
                          <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z" />
                        </svg>
                      </div>
                      Convolutional Layer Outputs
                    </CardTitle>
                    <p className="text-sm text-slate-500 mt-1">Network architecture with internal feature maps</p>
                  </CardHeader>
                  <CardContent>
                    {main.length > 0 ? (
                      <div className="overflow-x-auto pb-4">
                        <div className="flex gap-6 min-w-max">
                          {main.map(([name, data], index) => {
                            const layerInternals = internals[name] || [];
                            const hasInternals = layerInternals.length > 0;
                            
                            return (
                              <div 
                                key={name} 
                                className="flex flex-col items-center animate-in fade-in zoom-in-95"
                                style={{ animationDelay: `${700 + index * 100}ms` }}
                              >
                                {/* Layer Name Header */}
                                <div className="text-center mb-3">
                                  <h3 className="text-sm font-bold text-slate-800 mb-1">
                                    {name}
                                  </h3>
                                  <span className="text-[10px] text-slate-500 font-mono bg-orange-50 px-2 py-0.5 rounded border border-orange-200">
                                    {data.shape.join(" x ")}
                                  </span>
                                </div>

                                {/* Scrollable Container for Layer Content */}
                                <div className="flex flex-col items-center p-4 rounded-xl bg-linear-to-br from-slate-50 to-orange-50 border-2 border-orange-200 shadow-lg h-150 overflow-y-auto scrollbar-thin scrollbar-thumb-orange-300 scrollbar-track-orange-100">
                                  <div className="space-y-4">
                                    {/* Main Layer Visualization */}
                                    <div className="flex flex-col items-center">
                                      <div className="rounded-lg overflow-hidden border-2 border-orange-400 bg-white shadow-md w-32 hover:shadow-xl transition-shadow duration-300">
                                        <FeatureMap data={data.values} title="" />
                                      </div>
                                      <p className="text-[10px] text-slate-600 font-semibold mt-2 bg-white px-2 py-1 rounded border border-slate-200">
                                        Main Output
                                      </p>
                                    </div>

                                    {/* Internal Features Stacked Vertically */}
                                    {hasInternals && (
                                      <>
                                        <div className="w-full border-t-2 border-orange-300 pt-3 mt-2">
                                          <p className="text-xs font-bold text-center text-orange-700 mb-3 flex items-center justify-center gap-1">
                                            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                                              <path fillRule="evenodd" d="M5 3a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2V5a2 2 0 00-2-2H5zm9 4a1 1 0 10-2 0v6a1 1 0 102 0V7zm-3 2a1 1 0 10-2 0v4a1 1 0 102 0V9zm-3 3a1 1 0 10-2 0v1a1 1 0 102 0v-1z" clipRule="evenodd" />
                                            </svg>
                                            Internal Features
                                          </p>
                                        </div>
                                        
                                        <div className="flex flex-col items-center space-y-3">
                                          {layerInternals.map(([internalName, internalData], idx) => (
                                            <div 
                                              key={internalName} 
                                              className="group relative animate-in fade-in zoom-in-95"
                                              style={{ animationDelay: `${800 + idx * 50}ms` }}
                                            >
                                              <div className="rounded overflow-hidden border-2 border-slate-300 hover:border-orange-400 transition-all duration-300 hover:shadow-md bg-white w-32">
                                                <FeatureMap 
                                                  data={internalData.values} 
                                                  title="" 
                                                  internal={true} 
                                                />
                                              </div>
                                              <p className="text-center text-[9px] text-slate-600 font-mono mt-1 truncate w-32 px-1 bg-slate-50 rounded py-0.5">
                                                {internalName.split('.').pop()}
                                              </p>
                                              {/* Hover Tooltip */}
                                              <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-10">
                                                <div className="bg-slate-900 text-white text-[10px] px-2 py-1 rounded whitespace-nowrap font-mono shadow-lg">
                                                  {internalName}
                                                </div>
                                              </div>
                                            </div>
                                          ))}
                                        </div>
                                      </>
                                    )}
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-12 px-4">
                        <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-slate-100 mb-4">
                          <svg className="w-8 h-8 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
                          </svg>
                        </div>
                        <p className="text-slate-600 font-semibold mb-1">No layers detected</p>
                        <p className="text-xs text-slate-400">Total keys found: {allKeys.length}</p>
                      </div>
                    )}
                    <div className="w-full flex justify-center">
                        <ColorScale width={200} height={12} min={-1} max={1}/>
                    </div>
                  </CardContent>
                </Card>
              </div>

            </div>
          )}
        </div>
      </div>
    </div>
  );
}
/**
 * API Client library connecting Next.js frontend to FastAPI backend.
 */

export interface Top3Prediction {
  class_id: number;
  breed_name: string;
  display_name: string;
  animal_type: string;
  confidence: number;
}

export interface InferenceTime {
  detection_ms: number;
  classification_ms: number;
  gradcam_ms: number;
  total_ms: number;
}

export interface GradCAMOutput {
  target_class_id: number;
  display_name: string;
  overlay_base64: string;
  heatmap_base64?: string;
}

export interface PredictResponse {
  animal_type: string;
  animal_confidence: number;
  bounding_box: [number, number, number, number];
  predicted_breed: string;
  breed_confidence: number;
  top_3_predictions: Top3Prediction[];
  prediction_status: string;
  inference_time: InferenceTime;
  model_versions: Record<string, string>;
  gradcam_output?: GradCAMOutput;
  model_version?: string;
  species?: string;
  confidence?: number;
  latency?: number;
  status?: string;
  request_id?: string;
}

export interface BreedInfo {
  breed_name: string;
  display_name: string;
  animal_type: string;
  origin_region: string;
  description: string;
}

export interface BreedsListResponse {
  total_breeds: number;
  breeds: BreedInfo[];
}

export interface ModelInfoResponse {
  system_name: string;
  pipeline_version: string;
  supported_species: string[];
  supported_breeds_count: number;
  checkpoints_status: Record<string, boolean>;
  environment_info: Record<string, any>;
}

const API_BASE_URL = typeof window !== 'undefined' ? '' : 'http://localhost:8000';

export async function fetchHealthStatus() {
  const res = await fetch(`${API_BASE_URL}/api/health`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch health status');
  return res.json();
}

export async function fetchBreedsList(): Promise<BreedsListResponse> {
  const res = await fetch(`${API_BASE_URL}/api/breeds`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch breeds list');
  return res.json();
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const res = await fetch(`${API_BASE_URL}/api/model-info`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch model info');
  return res.json();
}

export async function predictBreed(
  imageFile: File,
  generateGradcam = true,
  modelVersion = 'selected_breeds_swin_v2'
): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append('file', imageFile);

  const url = `${API_BASE_URL}/api/predict?generate_gradcam=${generateGradcam}&model_version=${encodeURIComponent(modelVersion)}`;
  const res = await fetch(url, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({ detail: 'API request failed' }));
    throw new Error(errData.detail || `Prediction failed with status ${res.status}`);
  }

  return res.json();
}

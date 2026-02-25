const API =
  process.env.NEXT_PUBLIC_API ||
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  'http://localhost:8000';

export interface Rect { x1: number; y1: number; x2: number; y2: number; }

export interface ViewParams {
  // getView에 보낼 파라미터(서버 스펙에 맞게 확장)
  [key: string]: unknown;
}

export interface ViewData {
  image_base64: string;
  cad_xlim: [number, number];
  cad_ylim: [number, number];
  pixel_width: number;
  pixel_height: number;
  view_id?: string;
}

export interface AnalyzePayload {
  cad_xlim: [number, number];
  cad_ylim: [number, number];
  pixel_width: number;
  pixel_height: number;
  required_rects: Rect[];
  forbidden_rects: Rect[];
}


//export interface AnalyzeResult {
//  add_unconditional?: Array<{ x1: number; y1: number; x2: number; y2: number }>;
//  fixed_zero?: Rect[];
//  non_zero?: Rect[];
//}
export interface AnalyzeResult {
  add_unconditional: string[][];
  fixed_zero: string[];
  non_zero: string[];
  not_equal: string[][];
  must_equal: string[][];
  fixed_value: Record<string, number>;
  forbidden_zones: Array<Record<string, number>>;
}

// 추가 타입
export interface AnalyzeOccPayload {
  view_id: string;
  required_rects: Rect[];
  forbidden_rects: Rect[];
  h1p_box?: Rect | null;
  rev_forming_box?: Rect | null; // [수정] 정의 추가
}
export interface AnalyzeFullResult {
  add_unconditional: string[][];
  fixed_zero: string[];
  non_zero: string[];
  not_equal: string[][];
  must_equal: string[][];
  fixed_value: Record<string, number>;
  forbidden_zones: Array<Record<string, number>>;
  h1_p_components?: { base: number; y: number };
}

// OCC 업로드(프로그레스)
export function viewOccUploadWithProgress(
  file: File,
  { z_threshold = -45.2, z_outline = -38.2, width = 1200 }: { z_threshold?: number; z_outline?: number; width?: number } = {},
  onProgress: (pct: number) => void = () => {}
): Promise<ViewData> {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('z_threshold', String(z_threshold));
    fd.append('z_outline', String(z_outline));
    if (width != null) fd.append('width', String(width));

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API}/api/view-occ-upload`);
    xhr.upload.onprogress = (evt) => {
      if (evt.lengthComputable) onProgress(Math.round((evt.loaded * 100) / evt.total));
    };
    xhr.onreadystatechange = () => {
      if (xhr.readyState === 4) {
        if (xhr.status >= 200 && xhr.status < 300) {
          try { resolve(JSON.parse(xhr.responseText)); }
          catch { reject(new Error('invalid JSON response')); }
        } else reject(new Error(xhr.responseText || 'upload failed'));
      }
    };
    xhr.onerror = () => reject(new Error('network error'));
    xhr.send(fd);
  });
}

// OCC 분석
export async function analyzeOcc(payload: AnalyzeOccPayload): Promise<AnalyzeFullResult> {
  const res = await fetch(`${API}/api/analyze-occ`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error('analyze-occ failed');
  return res.json();
}

// h1_p 단독 계산(미리보기용 선택)
export async function h1pOcc(req: { view_id: string; box: Rect }): Promise<{ value: number; matched_y: number; base: number }> {
  const res = await fetch(`${API}/api/h1p-occ`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error('h1p-occ failed');
  return res.json();
}


export async function getView(params: ViewParams = {}): Promise<ViewData> {
  const res = await fetch(`${API}/api/view`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error('view failed');
  return res.json();
}

export async function analyze(payload: AnalyzePayload): Promise<AnalyzeResult> {
  const res = await fetch(`${API}/api/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error('analyze failed');
  return res.json();
}

export async function viewUpload(
  file: File,
  { z_threshold = -45.2, z_outline = -38.2, width = 1200 }: { z_threshold?: number; z_outline?: number; width?: number } = {}
): Promise<ViewData> {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('z_threshold', String(z_threshold));
  fd.append('z_outline', String(z_outline));
  if (width != null) fd.append('width', String(width));

  const res = await fetch(`${API}/api/view-upload`, {
    method: 'POST',
    body: fd,
  });
  if (!res.ok) throw new Error('view-upload failed');
  return res.json();
}

export function viewUploadWithProgress(
  file: File,
  { z_threshold = -45.2, z_outline = -38.2, width = 1200 }: { z_threshold?: number; z_outline?: number; width?: number } = {},
  onProgress: (pct: number) => void = () => {}
): Promise<ViewData> {
  return new Promise((resolve, reject) => {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('z_threshold', String(z_threshold));
    fd.append('z_outline', String(z_outline));
    if (width != null) fd.append('width', String(width));

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API}/api/view-upload`);

    xhr.upload.onprogress = (evt) => {
      if (evt.lengthComputable) {
        const pct = Math.round((evt.loaded * 100) / evt.total);
        onProgress(pct);
      }
    };

    xhr.onreadystatechange = () => {
      if (xhr.readyState === 4) {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch {
            reject(new Error('invalid JSON response'));
          }
        } else {
          reject(new Error(xhr.responseText || 'upload failed'));
        }
      }
    };

    xhr.onerror = () => reject(new Error('network error'));
    xhr.send(fd);
  });
}

export async function optimizeMask(payload: {
  add_unconditional?: string[][];
  fixed_zero?: string[];
  non_zero?: string[];
  not_equal?: string[][];
  fixed_value?: Record<string, number>;
  must_equal?: string[][];
  max_generations?: number;
  p_crossover?: number;
  p_mutation?: number;
  num_random_individuals?: number;
  early_stop_patience?: number;
  view_id?: string;   // 추가: 백엔드로 넘길 view_id
}): Promise<{ mask_base64: string; 
  overlay_base64: string;
  contour_base64?: string;
  min_fitness: number;
  max_displacement?: number;}> {
  const res = await fetch(`${API}/api/optimize-mask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error('optimize-mask failed');
  return res.json();
}

export async function getPreoptMaxDisp(view_id: string): Promise<{ max_displacement: number }> {
  const res = await fetch(`${API}/api/preopt-max-disp`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ view_id }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`preopt-max-disp failed: ${text}`);
  }
  return res.json();
}
"use client";

import { useState, useEffect } from "react";
import {
  mdiFileUpload,
  mdiChevronRight,
  mdiCheck,
  mdiInformation,
  mdiChevronDown,
  mdiHistory,
  mdiArrowLeft,
  mdiAlert,
} from "@mdi/js";
import Icon from "@mdi/react";
import LoadingSpinner from "@/components/LoadingSpinner";
import { useRouter } from "next/navigation";
import Link from "next/link";
import Toast from "@/components/Toast";
import { useRef } from "react";
import {
  //analyze,
  //viewUploadWithProgress,
  analyzeOcc,
  viewOccUploadWithProgress,
  optimizeMask,
  getPreoptMaxDisp,
  type Rect,
  type ViewData
} from "@/lib/coverBottomApi";

interface LEDHousingFormParams {
  material: string;
  pcb_length: number;
  led_housing_gap: number;
  led_spacing: number;
  led_count: number;
  led_watt: number;
  led_length_x: number;
  led_height_y: number;
  led_depth_z: number;
}

interface LEDHousingFormErrors {
  [key: string]: { type: string; message: string } | undefined;
}

// Constants
const COVER_BOTTOM_MODELS = ["UA Series", "UB Series", "Universal"];
const COVER_BOTTOM_MATERIALS = ["Aluminium", "EGI"];
const MAX_FILE_SIZE = 500 * 1024 * 1024; // 500MB

export default function LEDHousingSimulation() {
  const [currentStep, setCurrentStep] = useState(1);
  const [fileName, setFileName] = useState("");
  const [params, setParams] = useState<LEDHousingFormParams>({
    material: "Aluminium",
    pcb_length: 945,
    led_housing_gap: 6.5,
    led_spacing: 1.74,
    led_count: 216,
    led_watt: 0.25,
    led_length_x: 7,
    led_height_y: 0.8,
    led_depth_z: 2.0,
  });
  const [remark, setRemark] = useState("");
  const [errors, setErrors] = useState<LEDHousingFormErrors>({});
  const [showUpload, setShowUpload] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const router = useRouter();

  const [showToast, setShowToast] = useState(false);
  const [toastMessage, setToastMessage] = useState("");
  const [toastType, setToastType] = useState<"error" | "warning" | "success">(
    "error"
  );

  const [view, setView] = useState<ViewData | null>(null);
  //const [mode, setMode] = useState<"required" | "forbidden">("required");
  const [requiredRects, setRequiredRects] = useState<Rect[]>([]);
  const [forbiddenRects, setForbiddenRects] = useState<Rect[]>([]);
  const [result, setResult] = useState<any>(null);
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const drawing = useRef<Rect | null>(null);
  //const [isDragging, setIsDragging] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [overlays, setOverlays] = useState<any>(null);
  const previewRef = useRef<HTMLCanvasElement | null>(null);
  const [optimizing, setOptimizing] = useState(false);
  const [maskB64, setMaskB64] = useState<string | null>(null);
  const [optErr, setOptErr] = useState<string | null>(null);
  
  // 추가 상태
  const [mode, setMode] = useState<"h1p" | "rev_forming" | "required" | "forbidden">("required");
  const [h1pRects, setH1pRects] = useState<Rect[]>([]);
  const [revFormingRects, setRevFormingRects] = useState<Rect[]>([]); // [추가]
  const [showOptionalTools, setShowOptionalTools] = useState(false);

  function onDragOver(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(true);
  }
  function onDragLeave(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
  }
  async function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
    const f = (e.dataTransfer as DataTransfer)?.files?.[0] || null;
    if (!f) return;
    setFile(f);
  }

  useEffect(() => {
    const cvs = canvasRef.current;
    if (!cvs || !view) return;
    const ctx = cvs.getContext("2d");
    if (!ctx) return;
    const img = new Image();
    img.src = "data:image/png;base64," + view.image_base64;
    img.onload = () => {
      cvs.width = img.width;
      cvs.height = img.height;
      ctx.clearRect(0, 0, cvs.width, cvs.height);
      ctx.drawImage(img, 0, 0);
      drawRects(ctx);
    };
  }, [view, requiredRects, forbiddenRects, h1pRects, revFormingRects]); // [추가] 의존성 배열에 revFormingRects 추가

  useEffect(() => {
    if (!showModal || !view || !overlays) return;
    const cvs = previewRef.current;
    if (!cvs) return;
    const ctx = cvs.getContext("2d");
    if (!ctx) return;
    const img = new Image();
    img.src = "data:image/png;base64," + view.image_base64;
    img.onload = () => {
      cvs.width = img.width;
      cvs.height = img.height;
      ctx.clearRect(0, 0, cvs.width, cvs.height);
      ctx.drawImage(img, 0, 0);
      drawOverlaysOn(ctx);
    };
  }, [showModal, view, overlays]);

  //unction drawRects(ctx: CanvasRenderingContext2D) {
  //  ctx.lineWidth = 2;
  //  ctx.setLineDash([4, 4]);
  //  ctx.strokeStyle = "blue";
  //  requiredRects.forEach(r => {
  //    const x = Math.min(r.x1, r.x2);
  //    const y = Math.min(r.y1, r.y2);
  //    const w = Math.abs(r.x2 - r.x1);
  //    const h = Math.abs(r.y2 - r.y1);
  //    ctx.strokeRect(x, y, w, h);
  //  });
  //  ctx.strokeStyle = "red";
  //  forbiddenRects.forEach(r => {
  //    const x = Math.min(r.x1, r.x2);
  //    const y = Math.min(r.y1, r.y2);
  //    const w = Math.abs(r.x2 - r.x1);
  //    const h = Math.abs(r.y2 - r.y1);
  //    ctx.strokeRect(x, y, w, h);
  //  });
  //}

  // 드로잉(렌더) 함수에 h1_p, rev_forming 박스 표시 추가
  function drawRects(ctx: CanvasRenderingContext2D) {
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    
    // h1_p
    ctx.strokeStyle = "green";
    h1pRects.forEach(r => {
      const x = Math.min(r.x1, r.x2), y = Math.min(r.y1, r.y2);
      const w = Math.abs(r.x2 - r.x1), h = Math.abs(r.y2 - r.y1);
      ctx.strokeRect(x, y, w, h);
    });

    // [추가] rev_forming (주황색)
    ctx.strokeStyle = "orange";
    revFormingRects.forEach(r => {
      const x = Math.min(r.x1, r.x2), y = Math.min(r.y1, r.y2);
      const w = Math.abs(r.x2 - r.x1), h = Math.abs(r.y2 - r.y1);
      ctx.strokeRect(x, y, w, h);
    });

    // required
    ctx.strokeStyle = "blue";
    requiredRects.forEach(r => {
      const x = Math.min(r.x1, r.x2), y = Math.min(r.y1, r.y2);
      const w = Math.abs(r.x2 - r.x1), h = Math.abs(r.y2 - r.y1);
      ctx.strokeRect(x, y, w, h);
    });
    // forbidden
    ctx.strokeStyle = "red";
    forbiddenRects.forEach(r => {
      const x = Math.min(r.x1, r.x2), y = Math.min(r.y1, r.y2);
      const w = Math.abs(r.x2 - r.x1), h = Math.abs(r.y2 - r.y1);
      ctx.strokeRect(x, y, w, h);
    });
  }
  
  function drawOverlaysOn(ctx: CanvasRenderingContext2D) {
    if (!overlays) return;
    ctx.setLineDash([]);
    ctx.lineWidth = 2;
    if (overlays.non_zero_vrects) {
      ctx.strokeStyle = "#00cfdc";
      ctx.globalAlpha = 0.9;
      overlays.non_zero_vrects.forEach((r: Rect) => {
        const x = Math.min(r.x1, r.x2), y = Math.min(r.y1, r.y2);
        const w = Math.abs(r.x2 - r.x1), h = Math.abs(r.y2 - r.y1);
        ctx.strokeRect(x, y, w, h);
      });
    }
    if (overlays.fixed_zero_vrects) {
      ctx.strokeStyle = "#e11d48";
      overlays.fixed_zero_vrects.forEach((r: Rect) => {
        const x = Math.min(r.x1, r.x2), y = Math.min(r.y1, r.y2);
        const w = Math.abs(r.x2 - r.x1), h = Math.abs(r.y2 - r.y1);
        ctx.strokeRect(x, y, w, h);
      });
    }
    if (overlays.add_unconditional_lines) {
      ctx.strokeStyle = "#d946ef";
      overlays.add_unconditional_lines.forEach((l: Rect) => {
        ctx.beginPath();
        ctx.moveTo(l.x1, l.y1);
        ctx.lineTo(l.x2, l.y2);
        ctx.stroke();
      });
    }
    ctx.globalAlpha = 1;
  }
  
  function onMouseDown(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!view) return;
    const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    drawing.current = { x1: x, y1: y, x2: x, y2: y };
  }
  function onMouseMove(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!drawing.current) return;
    const cvs = canvasRef.current!;
    const rect = cvs.getBoundingClientRect();
    drawing.current.x2 = e.clientX - rect.left;
    drawing.current.y2 = e.clientY - rect.top;
    const ctx = cvs.getContext("2d")!;
    const img = new Image();
    img.src = "data:image/png;base64," + (view as ViewData).image_base64;
    img.onload = () => {
      ctx.clearRect(0, 0, cvs.width, cvs.height);
      ctx.drawImage(img, 0, 0);
      drawRects(ctx);

      const d = drawing.current;
      if (!d) return; 
  
      ctx.setLineDash([4, 4]);
      // [수정] 모드별 색상 지정
      if (mode === "h1p") ctx.strokeStyle = "green";
      else if (mode === "rev_forming") ctx.strokeStyle = "orange";
      else if (mode === "required") ctx.strokeStyle = "blue";
      else ctx.strokeStyle = "red"; // forbidden

      //const d = drawing.current!;
      const x = Math.min(d.x1, d.x2);
      const y = Math.min(d.y1, d.y2);
      const w = Math.abs(d.x2 - d.x1);
      const h = Math.abs(d.y2 - d.y1);
      ctx.strokeRect(x, y, w, h);
    };
  }

  function onMouseUp() {
    if (!drawing.current) return;
    const d = drawing.current;
    const rect = { x1: d.x1, y1: d.y1, x2: d.x2, y2: d.y2 };
    
    // [수정] 모드별 배열 추가 로직
    if (mode === "h1p") setH1pRects(prev => [...prev, rect]);
    else if (mode === "rev_forming") setRevFormingRects(prev => [...prev, rect]);
    else if (mode === "required") setRequiredRects(prev => [...prev, rect]);
    else setForbiddenRects(prev => [...prev, rect]);
    drawing.current = null;
  }
  
  
  //async function onViewUpload(f?: File) {
  //  const fileToUse = f ?? file ?? null;
  //  if (!fileToUse) { setErr("파일을 먼저 선택/드롭하세요."); return; }
  //  setUploading(true);
  //  setProgress(0);
  //  setErr(null);
  //  try {
  //    const v = await viewUploadWithProgress(
  //      fileToUse,
  //      { z_threshold: -45.2, z_outline: -38.2, width: 720},
  //      (p) => setProgress(p)
  //    );
  //    setView(v);
  //    setRequiredRects([]);
  //    setForbiddenRects([]);
  //    setResult(null);
  //  } catch (e: any) {
  //    setErr(e?.message || String(e));
  //  } finally {
  //    setUploading(false);
  //  }
  //}

  // 업로드 핸들러 교체
  async function onViewUpload(f?: File) {
    const fileToUse = f ?? file ?? null;
    if (!fileToUse) { setErr("파일을 먼저 선택/드롭하세요."); return; }
    setUploading(true); setProgress(0); setErr(null);
    try {
      const v = await viewOccUploadWithProgress(
        fileToUse,
        { z_threshold: -45.2, z_outline: -38.2, width: 720 },
        (p) => setProgress(p)
      );
      setView(v);
      setH1pRects([]);
      setRevFormingRects([]); // [추가] 초기화
      setRequiredRects([]);
      setForbiddenRects([]);
      setResult(null);
    } catch (e: any) {
      setErr(e?.message || String(e));
    } finally {
      setUploading(false);
    }
  }
  
  //async function onAnalyze() {
  //  if (!view) return;
  //  setShowModal(true);
  //  setOptimizing(true);
  //  setMaskB64(null);
  //  setOptErr(null);
  //  const payload = {
  //    cad_xlim: view.cad_xlim,
  //    cad_ylim: view.cad_ylim,
  //    pixel_width: view.pixel_width,
  //    pixel_height: view.pixel_height,
  //    required_rects: requiredRects,
  //    forbidden_rects: forbiddenRects,
  //  };
  //  try {
  //    const res = await analyze(payload);
      // 백엔드가 overlays도 반환한다면 setOverlays(res) 등으로 사용
  //    const opt = await optimizeMask({
  //      add_unconditional: res.add_unconditional || [],
  //      fixed_zero: res.fixed_zero || [],
  //      non_zero: res.non_zero || [],
  //      // max_generations: 30, num_random_individuals: 120,
  //      view_id: view.view_id,
  //    });
  //    setMaskB64(opt.mask_base64);
  //  } catch (e: any) {
  //    setOptErr(e?.message || String(e));
  //  } finally {
  //    setOptimizing(false);
  //  }
  //} 

  async function onAnalyze() {
    if (!view) return;
    setShowModal(true);
    setOptimizing(true);
    setMaskB64(null);
    setOptErr(null);
  
    try {
      const res = await analyzeOcc({
        view_id: view.view_id!,
        required_rects: requiredRects,
        forbidden_rects: forbiddenRects,
        h1p_box: h1pRects.length > 0 ? h1pRects[0] : undefined,
        // [추가] rev_forming 박스 전송 (첫 번째 박스 사용)
        rev_forming_box: revFormingRects.length > 0 ? revFormingRects[0] : undefined, 
      });
  
      const opt = await optimizeMask({
        add_unconditional: res.add_unconditional || [],
        fixed_zero: res.fixed_zero || [],
        non_zero: res.non_zero || [],
        not_equal: res.not_equal || [],
        must_equal: res.must_equal || [],
        fixed_value: res.fixed_value || {}, // h1_p 포함
        upper_bound: res.upper_bound || {},
        view_id: view.view_id,
      });
  
      setMaskB64(opt.mask_base64);
    } catch (e: any) {
      setOptErr(e?.message || String(e));
    } finally {
      setOptimizing(false);
    }
  }

  function onClear() {
    setH1pRects([]);
    setRevFormingRects([]); // [추가] 초기화
    setRequiredRects([]);
    setForbiddenRects([]);
    setResult(null);
  }
  
  useEffect(() => {
    const timer = setTimeout(() => {
      setShowUpload(true);
    }, 500);

    return () => clearTimeout(timer);
  }, []);

  const validateFileSize = (size: number, maxSize: number): string | null => {
    if (size > maxSize) {
      return `파일 크기가 너무 큽니다. 최대 ${Math.floor(maxSize / (1024 * 1024))}MB까지 업로드 가능합니다.`;
    }
    return null;
  };

  const validateFileExtension = (fileName: string): string | null => {
    const allowedExtensions = ['.prt', '.stp'];
    const extension = fileName.toLowerCase().substring(fileName.lastIndexOf('.'));
    if (!allowedExtensions.includes(extension)) {
      return '지원되는 파일 형식: .prt, .stp';
    }
    return null;
  };

  const handleParamChange = (name: string, value: string | number) => {
    setParams((prev) => ({ ...prev, [name]: value }));
    setError(name, undefined);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
  
    const sizeError = validateFileSize(file.size, MAX_FILE_SIZE);
    if (sizeError) {
      setError("file", { type: "manual", message: sizeError });
      e.target.value = "";
      return;
    }
  
    const extensionError = validateFileExtension(file.name);
    if (extensionError) {
      setError("file", { type: "manual", message: extensionError });
      e.target.value = "";
      return;
    }
  
    setError("file", undefined);
    setFileName(file.name);
    setFile(file);
    setCurrentStep(2);
    void onViewUpload(file); 
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
  
    const extensionError = validateFileExtension(file.name);
    if (extensionError) {
      setError("file", { type: "manual", message: extensionError });
      return;
    }
  
    const sizeError = validateFileSize(file.size, MAX_FILE_SIZE);
    if (sizeError) {
      setError("file", { type: "manual", message: sizeError });
      return;
    }
  
    const fileInput = document.getElementById("fileInput") as HTMLInputElement;
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    fileInput.files = dataTransfer.files;
  
    setError("file", undefined);
    setFileName(file.name);
    setFile(file);
    setCurrentStep(2);
    void onViewUpload(file); // 드롭 즉시 영역 지정용 미리보기 업로드
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const isFormValid = () => {
    return params.material !== "" && Object.keys(errors).length === 0;
  };

  const showToastMessage = (
    message: string,
    type: "error" | "warning" | "success" = "error"
  ) => {
    setToastMessage(message);
    setToastType(type);
    setShowToast(true);
  };

  const handleSimulationStart = async () => {
    try {
      if (!isFormValid()) return;
      if (!view) {
        showToastMessage("먼저 STP를 업로드해 영역을 지정하세요", "warning");
        return;
      }
      setIsUploading(true);
      const pre = await getPreoptMaxDisp(view.view_id!);
  
      //const payload = {
      //  cad_xlim: view.cad_xlim,
      //  cad_ylim: view.cad_ylim,
      //  pixel_width: view.pixel_width,
      //  pixel_height: view.pixel_height,
      //  required_rects: requiredRects,
      //  forbidden_rects: forbiddenRects,
      //};

      const requestedAt = new Date().toISOString();
      //const res = await analyze(payload);
      const res = await analyzeOcc({
        view_id: view.view_id!,
        required_rects: requiredRects,
        forbidden_rects: forbiddenRects,
        h1p_box: h1pRects.length > 0 ? h1pRects[0] : undefined,
      });
  
      
      const startedAt = new Date().toISOString();

      //const opt = await optimizeMask({
      //  add_unconditional: res.add_unconditional || [],
      //  fixed_zero: res.fixed_zero || [],
      //  non_zero: res.non_zero || [],
      //  view_id: view.view_id,
      //});
      const opt = await optimizeMask({
        add_unconditional: res.add_unconditional || [],
        fixed_zero: res.fixed_zero || [],
        non_zero: res.non_zero || [],
        not_equal: res.not_equal || [],
        must_equal: res.must_equal || [],
        fixed_value: res.fixed_value || {},
        upper_bound: res.upper_bound || {},
        view_id: view.view_id,
      });


      const completedAt = new Date().toISOString();
  
      sessionStorage.setItem("cover_opt_result", JSON.stringify({
        overlay_base64: opt.overlay_base64,
        contour_base64: opt.contour_base64,
        mask_base64: opt.mask_base64,
        preopt_max_displacement: pre.max_displacement,
        max_displacement: opt.max_displacement,
        metadata: {
          file_name: fileName || "CoverBottom.stp",
          remark,
          create_date: requestedAt,
          analysis_start_date: startedAt,
          complete_date: completedAt,
        },
        parameters: { material: params.material },
      }));
  
      showToastMessage("최적화 결과가 생성되었습니다", "success");
      router.push("/simulation/cover_bottom/result");
    } catch (error) {
      console.error("Optimization failed:", error);
      const errorMessage = error instanceof Error ? error.message : "알 수 없는 오류가 발생했습니다";
      showToastMessage(errorMessage, "error");
    } finally {
      setIsUploading(false);
    }
  };

  const setError = (
    field: string,
    error: { type: string; message: string } | undefined
  ) => {
    setErrors((prev) => {
      const newErrors = { ...prev };
      if (error) {
        newErrors[field] = error;
      } else {
        delete newErrors[field];
      }
      return newErrors;
    });
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      {isUploading && <LoadingSpinner />}
      {showToast && (
        <Toast
          message={toastMessage}
          type={toastType}
          onClose={() => setShowToast(false)}
        />
      )}

      <div className="max-w-4xl mx-auto">
        <div className="fade-in">
          <h1 className="text-3xl font-bold text-gray-800 mb-4">
            Cover Bottom 최적화 Simulation
          </h1>

          <div className="flex items-center gap-4 mb-6">
            <Link
              href="/simulation"
              className="inline-flex items-center gap-2 px-4 py-2 bg-white text-[#A50034] border border-[#A50034] rounded-md hover:bg-[#A50034] hover:text-white transition-colors"
            >
              <Icon path={mdiArrowLeft} size={0.9} />
              Go Back
            </Link>
            <Link
              href="/simulation/cover_bottom/queue"
              className="inline-flex items-center gap-2 px-4 py-2 bg-white text-[#A50034] border border-[#A50034] rounded-md hover:bg-[#A50034] hover:text-white transition-colors"
            >
              <Icon path={mdiHistory} size={0.9} />
              Simulation History
            </Link>
            <Link
              href="/simulation/cover_bottom/release"
              className="inline-flex items-center gap-2 px-4 py-2 bg-white text-[#A50034] border border-[#A50034] rounded-md hover:bg-[#A50034] hover:text-white transition-colors"
            >
              <Icon path={mdiInformation} size={0.9} />
              Release Note
            </Link>
          </div>

          <div>
            <div className="bg-gradient-to-r from-[#A50034] to-[#8A002C] rounded-lg p-[1px]">
              <div className="bg-white rounded-lg p-6">
                <div className="flex items-start space-x-4">
                  <div className="flex-shrink-0 w-1 h-16 bg-[#A50034] rounded-full"></div>
                  <div>
                    <p className="text-lg text-gray-700 leading-relaxed">
                      <span className="font-semibold text-[#A50034]">
                        STP CAD 파일
                      </span>{" "}
                      Input 만으로 Cover Bottom의 최적 강성 구조 도출이 가능합니다.
                      <br />
                      복잡한 해석 과정 없이,{" "}
                      <span className="font-semibold text-[#A50034]">
                        자동화된 시뮬레이션
                      </span>
                      으로 최적 구조를 즉시 확인하세요.
                      <br />
                      <span className="text-sm text-amber-600 font-medium">
                        ※ 현재 Cover Bottom 단품 형상만 지원합니다. 다른 Press 부품 입력 시 오류가 발생할 수 있습니다.
                      </span>
                      <br />
                      <span className="text-sm text-red-500 font-medium">
                        ※ PEMNUT이나 방열판 등이 Assy된 형상은 지원하지 않습니다.
                      </span>
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 파일 업로드 섹션 */}
        {showUpload && (
          <div
            className={`mb-8 transition-all duration-500 delayed-fade-in
            ${currentStep === 1 ? "" : "opacity-50"}`}
          >
            <div
              className={`bg-white rounded-lg shadow-lg p-8 border-2 border-dashed 
                ${
                  errors.file
                    ? "border-red-500 bg-red-50"
                    : isDragging
                    ? "border-[#A50034] bg-red-50"
                    : "border-gray-300 hover:border-[#A50034]"
                } 
                transition-all duration-200 cursor-pointer`}
              onClick={() => document.getElementById("fileInput")?.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
            >
              <div className="text-center">
                <Icon
                  path={fileName ? mdiCheck : mdiFileUpload}
                  size={2}
                  className={`mx-auto mb-4 ${
                    errors.file
                      ? "text-red-500"
                      : fileName
                      ? "text-green-500"
                      : isDragging
                      ? "text-[#A50034]"
                      : "text-gray-400"
                  }`}
                />
                <input
                  id="fileInput"
                  type="file"
                  accept=".prt,.stp"
                  onChange={handleFileChange}
                  className="hidden"
                />
                <span
                  className={`text-lg font-medium ${
                    errors.file ? "text-red-600" : "text-gray-700"
                  }`}
                >
                  {fileName || "Upload Cover Bottom STP File"}
                </span>
                {!fileName && !errors.file && (
                  <p className="mt-2 text-sm text-gray-500">
                    {isDragging
                      ? "Drop your STP file here"
                      : "Click or drag and drop your STP file here (Max: 500MB)"}
                  </p>
                )}
                {errors.file && (
                  <p className="mt-2 text-sm text-red-600 bg-red-100 p-2 rounded">
                    {errors.file.message}
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* 매개변수 입력 섹션 */}
        {currentStep === 2 && (
          <div className="transition-all duration-500 fade-in space-y-8">
            
            {/* Card 1: Basic Information */}
            <div className="bg-white rounded-lg shadow-lg p-8">
              <h3 className="text-sm font-semibold text-gray-800 mb-6 flex items-center gap-2">
                영역 지정을 해주세요.
              </h3>
              {/* 영역 지정 업로드/미리보기/도구 */}
              <div className="space-y-4">
                {uploading && (
                  <div className="mt-2 w-80 h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div className="h-2 bg-cyan-600 transition-[width] duration-100" style={{ width: `${Math.min(progress, 100)}%` }} />
                  </div>
                )}
                {err && <div className="text-red-600 text-sm">에러: {err}</div>}

                {/* 툴/모드 */}
                {view && (
                  <>
                    <div className="flex flex-col gap-2">
                      <div className="flex items-center gap-2">
                        <button onClick={() => setMode("required")} className={`px-3 py-2 rounded-md border ${mode === "required" ? "bg-blue-300 border-blue-400" : "bg-white border-gray-300"}`}>
                        필수 포밍 존재 구역
                        </button>
                        <button onClick={() => setMode("forbidden")} className={`px-3 py-2 rounded-md border ${mode === "forbidden" ? "bg-rose-300 border-red-400" : "bg-white border-gray-300"}`}>
                        포밍 불가 구역
                        </button>
                        <button onClick={onClear} className="px-3 py-2 rounded-md border bg-white">
                        초기화
                        </button>
                        
                        <div className="flex-grow"></div> 

                        {/* 선택사항 체크박스 삭제 희망 시 이 아래 전부 주석 처리 */}
                        <label className="flex items-center gap-2 cursor-pointer bg-white px-3 py-2 rounded-md border border-gray-300 select-none">
                            <input 
                                type="checkbox" 
                                checked={showOptionalTools} 
                                onChange={(e) => setShowOptionalTools(e.target.checked)} 
                                className="w-4 h-4 text-[#A50034] rounded focus:ring-[#A50034]"
                            />
                            <span className="text-sm">선택 사항</span>
                        </label>
                        {/* 여기까지 */}

                      </div>

                      {showOptionalTools && (
                        <div className="flex items-center gap-2 p-2 bg-gray-50 rounded border border-gray-200">
                            <button onClick={() => setMode("h1p")} className={`px-3 py-2 text-sm rounded-md border ${mode === "h1p" ? "bg-green-300 border-green-400" : "bg-white border-gray-300"}`}>
                            h1_p 추출 박스
                            </button>
                            <button onClick={() => setMode("rev_forming")} className={`px-3 py-2 text-sm rounded-md border ${mode === "rev_forming" ? "bg-orange-300 border-orange-400" : "bg-white border-gray-300"}`}>
                            rev_forming 박스
                            </button>
                        </div>
                      )}
                    </div>

                    {/* 캔버스 */}
                    <canvas
                      ref={canvasRef}
                      onMouseDown={onMouseDown}
                      onMouseMove={onMouseMove}
                      onMouseUp={onMouseUp}
                      className="border border-gray-300 bg-white cursor-crosshair"
                    />
                  </>
                )}

                {/* 결과 마스크 모달 */}
                {showModal && (
                  <div
                    className="fixed inset-0 bg-black/40 flex items-center justify-center z-[9999]"
                    onClick={() => setShowModal(false)}
                  >
                    <div
                      className="bg-white p-4 rounded-md max-w-[90vw] max-h-[90vh]"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <div className="mb-2 font-semibold">최적 형상 마스크</div>
                      {optimizing && <div className="p-4">최적화/시각화 처리 중...</div>}
                      {!optimizing && optErr && <div className="text-red-600">에러: {optErr}</div>}
                      {!optimizing && maskB64 && (
                        <img
                          alt="mask"
                          src={`data:image/png;base64,${maskB64}`}
                          className="max-w-[80vw] max-h-[70vh] border border-gray-300"
                        />
                      )}
                      <div className="mt-2 text-right">
                        <button onClick={() => setShowModal(false)} className="px-3 py-2 rounded-md border bg-white">닫기</button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
                <span className="text-sm text-gray-800 font-medium">
                  ※ 필수 포밍 존재 구역 및 포밍 불가 구역을 마우스로 지정해주세요.
                </span>
                <br />
                <span className="text-sm text-gray-800 font-medium">
                  ※ 해당 구역 마우스 좌클릭 드래그하여 지정
                  <br />
                  - 필수 포밍 존재 구역은 파란색, 포밍 불가 구역은 빨간색
                  <br />
                </span>
              <br />
              <h3 className="text-sm font-semibold text-gray-800 mb-6 flex items-center gap-2">
                매개변수들을 입력해주세요.
              </h3>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* LED Housing 재질 */}
                <div>
                  <label className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
                    Model
                  </label>
                  <div className="relative">
                    <select
                      value={params.material}
                      onChange={(e) => handleParamChange("material", e.target.value)}
                      className={`w-full px-4 py-3 border rounded-md appearance-none bg-white
                      focus:ring-2 focus:ring-[#A50034] focus:border-transparent
                      ${errors.material ? "border-red-500" : "border-gray-300"}
                      cursor-pointer transition-all duration-200
                      hover:border-[#A50034] text-gray-700 font-medium`}
                    >
                      <option value="" disabled className="text-gray-500">
                        Select Model Type
                      </option>
                      {COVER_BOTTOM_MODELS.map((material) => (
                        <option
                          key={material}
                          value={material}
                          className="py-2 font-medium"
                        >
                          {material}
                        </option>
                      ))}
                    </select>
                    <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
                      <Icon
                        path={mdiChevronDown}
                        size={1}
                        className="text-gray-400"
                      />
                    </div>
                  </div>
                  {!params.material && (
                    <p className="mt-1 text-sm text-[#A50034] flex items-center gap-1">
                      <Icon path={mdiInformation} size={0.7} />
                      Material type is required
                    </p>
                  )}
                </div>
              </div>
            </div>

            {/* Remark */}
            <div className="bg-white rounded-lg shadow-lg p-8">
              {/* LED Housing 재질 */}
              <div>
                  <label className="text-sm font-medium text-gray-700 mb-2 flex items-center gap-2">
                    Material
                    <div className="relative group">
                      <Icon
                        path={mdiInformation}
                        size={0.8}
                        className="text-gray-500 hover:text-[#A50034] transition-colors"
                      />
                      <div className="absolute z-10 invisible group-hover:visible bg-gray-900 text-white text-sm rounded py-1 px-2 right-0 bottom-full mb-2 whitespace-nowrap">
                        Select material type
                        <div className="absolute top-full right-4 -mt-1 border-4 border-transparent border-t-gray-900"></div>
                      </div>
                    </div>
                  </label>
                  <div className="relative">
                    <select
                      value={params.material}
                      onChange={(e) => handleParamChange("material", e.target.value)}
                      className={`w-full px-4 py-3 border rounded-md appearance-none bg-white
                      focus:ring-2 focus:ring-[#A50034] focus:border-transparent
                      ${errors.material ? "border-red-500" : "border-gray-300"}
                      cursor-pointer transition-all duration-200
                      hover:border-[#A50034] text-gray-700 font-medium`}
                    >
                      <option value="" disabled className="text-gray-500">
                        Select Material Type
                      </option>
                      {COVER_BOTTOM_MATERIALS.map((material) => (
                        <option
                          key={material}
                          value={material}
                          className="py-2 font-medium"
                        >
                          {material}
                        </option>
                      ))}
                    </select>
                    <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
                      <Icon
                        path={mdiChevronDown}
                        size={1}
                        className="text-gray-400"
                      />
                    </div>
                  </div>
                  {!params.material && (
                    <p className="mt-1 text-sm text-[#A50034] flex items-center gap-1">
                      <Icon path={mdiInformation} size={0.7} />
                      Material type is required
                    </p>
                  )}
              </div>
              <br />
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Remark
              </label>
              <textarea
                className={`w-full px-4 py-2 border rounded-md focus:ring-2 focus:ring-[#A50034] focus:border-transparent border-gray-300`}
                value={remark}
                onChange={(e) => setRemark(e.target.value)}
                placeholder="Enter any additional notes or comments"
                rows={4}
              />
            </div>

            {/* Start Simulation Button */}
            <button
              className="w-full bg-[#A50034] text-white py-3 px-6 rounded-md hover:bg-[#8A002C] transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              onClick={handleSimulationStart}
              disabled={!isFormValid() || isUploading}
            >
              {isUploading ? "Processing..." : "Start Optimization"}
              <Icon path={mdiChevronRight} size={1} />
            </button>
          </div>
        )}
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .fade-in {
          animation: fadeIn 0.5s ease-out;
        }
        .delayed-fade-in {
          opacity: 0;
          animation: fadeIn 0.5s ease-out forwards;
          animation-delay: 0.5s;
        }
      `}</style>
    </div>
  );
}
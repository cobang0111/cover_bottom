"use client";

import { useState, useEffect } from "react";
import { useRef } from "react";
import { useRouter } from "next/navigation";
import Icon from "@mdi/react";
import {
  mdiArrowLeft,
  mdiInformation,
  mdiCheckCircle,
  mdiAlertCircle,
  mdiCog,
  mdiDownload,
  mdiTextBox,
  mdiChartBox,
  mdiThermometer,
} from "@mdi/js";
import LoadingSpinner from "@/components/LoadingSpinner";
import Image from "next/image";
import { formatDateToKST } from "@/utils/kst_date";

interface LEDHousingResultData {
  metadata: {
    file_name: string;
    status: "completed" | "failed" | "processing";
    user_name: string;
    user_group: string;
    create_date: string;
    analysis_start_date?: string;
    complete_date?: string;
    remark?: string;
    error?: string;
  };
  parameters: {
    material: string;
    pcb_length: number;
    led_housing_gap: number;
    led_spacing: number;
    led_count: number;
    led_watt: number;
    led_length_x: number;
    led_height_y: number;
    led_depth_z: number;
  };
  results: {
    overlay_image?: string;
    contour_image?: string;
    combined_image?: string;
    max_displacement?: number;
    preopt_max_displacement?: number;
    hotspots: Array<{
      no: number;
      x: number;
      y: number;
      z: number;
      temperature: number;
      spec: number;
      judge: "OK" | "NG";
    }>;
  };
}

export default function LEDHousingSimulationResult() {
  const [result, setResult] = useState<LEDHousingResultData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  const processed = useRef(false);

  useEffect(() => {
    if (processed.current) return;
    processed.current = true;
    try {
      const raw = sessionStorage.getItem("cover_opt_result");
      if (!raw) { setError("결과 데이터가 없습니다. 최적화를 다시 실행해 주세요."); setLoading(false); return; }
      const data = JSON.parse(raw);
      const mapped: LEDHousingResultData = {
        metadata: {
          file_name: data?.metadata?.file_name || "CoverBottom.stp",
          status: "completed",
          user_name: "user",
          user_group: "TV CAE팀",
          create_date: data?.metadata?.create_date || new Date().toISOString(),
          analysis_start_date: data?.metadata?.analysis_start_date,
          complete_date: data?.metadata?.complete_date,
          remark: data?.metadata?.remark || "",
        },
        parameters: {
          material: data?.parameters?.material || "",
          pcb_length: 0, led_housing_gap: 0, led_spacing: 0, led_count: 0,
          led_watt: 0, led_length_x: 0, led_height_y: 0, led_depth_z: 0,
        },
        results: {
          overlay_image: data?.overlay_base64
            ? `data:image/png;base64,${data.overlay_base64}`
            : (data?.mask_base64 ? `data:image/png;base64,${data.mask_base64}` : undefined),
          contour_image: data?.contour_base64
            ? `data:image/png;base64,${data.contour_base64}`
            : undefined,
          combined_image: data?.combined_base64
            ? `data:image/png;base64,${data.combined_base64}`
            : undefined,
          max_displacement: data?.max_displacement,
          preopt_max_displacement: data?.preopt_max_displacement,
          hotspots: [],
        },
      };
      setResult(mapped);
      // sessionStorage.removeItem("cover_opt_result");  // ← 삭제하지 않음
    } catch {
      setError("결과 파싱 실패");
    } finally {
      setLoading(false);
    }
  }, []);

  const handleDownload = async () => {
    try {
      const link = document.createElement("a");
      link.href = result?.results.contour_image || "";
      link.setAttribute("download", `${result?.metadata.file_name || "result"}.png`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      console.error("Failed to download file:", err);
    }
  };

  if (loading) return <LoadingSpinner />;
  if (error)
    return (
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="max-w-5xl mx-auto">
          <div className="bg-red-50 border border-red-200 rounded-lg p-6">
            <div className="flex items-start gap-4">
              <Icon
                path={mdiAlertCircle}
                size={1.5}
                className="text-red-500 flex-shrink-0"
              />
              <div>
                <h3 className="text-lg font-semibold text-red-800 mb-2">
                  오류 발생
                </h3>
                <p className="text-red-600">{error}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  if (!result) return null;

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-8 flex items-center justify-between">
          <button
            onClick={() => router.back()}
            className="flex items-center text-gray-600 hover:text-[#A50034] transition-colors"
          >
            <Icon path={mdiArrowLeft} size={1} className="mr-2" />
            뒤로 가기
          </button>
          <h1 className="text-2xl font-bold text-gray-800">
            Cover Bottom 최적화 결과
          </h1>
          <div className="w-24"></div>
        </div>

        {/* Status Card */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2
                className="text-xl font-semibold text-gray-800 mb-2 flex items-center gap-2 hover:text-[#A50034] cursor-pointer"
                onClick={handleDownload}
              >
                <Icon path={mdiDownload} size={1} />
                {result.metadata.file_name}
              </h2>
              <div className="space-y-1">
                <p className="text-sm text-gray-500">
                  요청자: {result.metadata.user_name} (
                  {result.metadata.user_group})
                </p>
                <p className="text-sm text-gray-500">
                  요청 생성 시간: {formatDateToKST(result.metadata.create_date)}
                </p>
                {result.metadata.analysis_start_date && (
                  <p className="text-sm text-gray-500">
                    해석 시작 시간:{" "}
                    {formatDateToKST(result.metadata.analysis_start_date)}
                  </p>
                )}
                {result.metadata.complete_date && (
                  <p className="text-sm text-gray-500">
                    완료 시간: {formatDateToKST(result.metadata.complete_date)}
                  </p>
                )}
              </div>
            </div>
            <div
              className={`px-4 py-2 rounded-full ${
                result.metadata.status === "completed"
                  ? "bg-green-100 text-green-800"
                  : result.metadata.status === "failed"
                  ? "bg-red-100 text-red-800"
                  : "bg-blue-100 text-blue-800"
              }`}
            >
              <div className="flex items-center gap-2">
                <Icon
                  path={
                    result.metadata.status === "completed"
                      ? mdiCheckCircle
                      : mdiInformation
                  }
                  size={0.8}
                />
                <span className="font-medium">
                  {result.metadata.status === "completed"
                    ? "완료"
                    : result.metadata.status === "failed"
                    ? "실패"
                    : "진행 중"}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Parameters Card */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <div className="flex gap-x-2 mb-4">
            <Icon path={mdiCog} size={1} className="text-[#A50034]" />
            <h3 className="text-lg font-semibold text-gray-800">
              입력 파라미터
            </h3>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            <div>
              <p className="text-sm text-gray-500 mb-1">재질</p>
              <p className="font-medium">{result.parameters.material}</p>
            </div>
          </div>
        </div>

        {/* Remark Card */}
        {result.metadata.remark && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
            <div className="flex gap-x-2 mb-4">
              <Icon path={mdiTextBox} size={1} className="text-[#A50034]" />
              <h3 className="text-lg font-semibold text-gray-800">
                Remark
              </h3>
            </div>
            <div className="bg-gray-50 rounded-lg p-4">
              <p className="text-gray-700 whitespace-pre-wrap">
                {result.metadata.remark}
              </p>
            </div>
          </div>
        )}

        {/* Results Card */}
        {result.metadata.status === "completed" && result.results && (
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex gap-x-2 mb-6">
              <Icon path={mdiChartBox} size={1} className="text-[#A50034]" />
              <h3 className="text-lg font-semibold text-gray-800">최적화 결과</h3>
            </div>

            {result.results.max_displacement && (
              <div className="mb-8">
                <p className="text-sm text-[#A50034]">최대 처짐량: {result.results.preopt_max_displacement.toFixed(2)}mm (기존) → {result.results.max_displacement.toFixed(2)}mm (최적화)</p>
              </div>
            )}

            {/* combined 이미지가 있으면 한 장 출력 */}
            {result.results.combined_image && (
              <div className="mb-8">
                <div className="border rounded-lg p-4 bg-gray-50">
                  <Image
                    src={result.results.combined_image}
                    alt="Optimization Overlay + Contour"
                    width={800}
                    height={600}
                    className="w-full rounded-lg"
                    unoptimized
                  />
                </div>
              </div>
            )}

            {!result.results.combined_image && (
              <div className="flex flex-col space-y-8">
                {result.results.overlay_image && (
                  <div>
                    <div className="border rounded-lg p-4 bg-gray-50">
                      <Image
                        src={result.results.overlay_image}
                        alt="Optimization Overlay"
                        width={800}
                        height={600}
                        className="w-full rounded-lg"
                        unoptimized
                      />
                    </div>
                  </div>
                )}

                {result.results.contour_image && (
                  <div>
                    <div className="border rounded-lg p-4 bg-gray-50">
                      <Image
                        src={result.results.contour_image}
                        alt="Displacement Contour"
                        width={800}
                        height={600}
                        className="w-full rounded-lg"
                        unoptimized
                      />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Error Message */}
        {result.metadata.status === "failed" && result.metadata.error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-6">
            <div className="flex items-start gap-4">
              <Icon
                path={mdiAlertCircle}
                size={1.5}
                className="text-red-500 flex-shrink-0"
              />
              <div>
                <h3 className="text-lg font-semibold text-red-800 mb-2">
                  시뮬레이션 실패
                </h3>
                <p className="text-red-600">{result.metadata.error}</p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

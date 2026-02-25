const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';

export async function startLedHousingSimulation(form: FormData) {
  const res = await fetch(`${API_BASE}/api/led-housing/simulations`, {
    method: 'POST',
    body: form
  });
  if (!res.ok) throw new Error('시뮬레이션 요청 실패');
  return res.json() as Promise<{ job_id: string }>;
}

export async function getLedHousingResult(id: string) {
  const res = await fetch(`${API_BASE}/api/led-housing/simulations/${id}`);
  if (!res.ok) throw new Error('결과 조회 실패');
  return res.json();
}
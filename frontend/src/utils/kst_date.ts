export function formatDateToKST(date: Date | string): string {
    let inputDate = date;
    if (typeof date === 'string' && !/(Z|[+-]\d{2}:?\d{2})$/.test(date)) {
        inputDate = date + 'Z';
    }
    const d = new Date(inputDate);
    
    if (isNaN(d.getTime())) {
        return '유효하지 않은 날짜입니다.';
    }

    const kstTimestamp = d.getTime() + 9 * 60 * 60 * 1000;
    const kstDate = new Date(kstTimestamp);
    const year = kstDate.getUTCFullYear();
    const month = String(kstDate.getUTCMonth() + 1).padStart(2, '0');
    const day = String(kstDate.getUTCDate()).padStart(2, '0');
    const hours = String(kstDate.getUTCHours()).padStart(2, '0');
    const minutes = String(kstDate.getUTCMinutes()).padStart(2, '0');
    const seconds = String(kstDate.getUTCSeconds()).padStart(2, '0');

    return `${year}-${month}-${day} ${hours}:${minutes}:${seconds} (KST)`;
}

export const convertToDateKST = (input: string) => {
    if(typeof input !== 'string') return
    const utcDate: Date = new Date(input);
    const kstOffset: number = 9 * 60 * 60 * 1000; 
    const kstDate: Date = new Date(utcDate.getTime() + kstOffset);
    return kstDate.toISOString().substring(0, 10); 
}

export const convertToTimeKST = (input: string) => {
    if(typeof input !== 'string') return
    const utcDate: Date = new Date(input);
    const kstOffset: number = 9 * 60 * 60 * 1000; 
    const kstDate: Date = new Date(utcDate.getTime() + kstOffset);
    return kstDate.toISOString().substring(11, 16); 
}

export const dateTimeKSTNow = () => {
    const utcDate: Date = new Date();
    const kstOffset: number = 9 * 60 * 60 * 1000; 
    const kstDate: Date = new Date(utcDate.getTime() + kstOffset);

    // Format date parts
    const year = kstDate.getUTCFullYear();
    const month = String(kstDate.getUTCMonth() + 1).padStart(2, '0');
    const day = String(kstDate.getUTCDate()).padStart(2, '0');
    const hours = String(kstDate.getUTCHours()).padStart(2, '0');
    const minutes = String(kstDate.getUTCMinutes()).padStart(2, '0');
    const seconds = String(kstDate.getUTCSeconds()).padStart(2, '0');
    
    return `${year}${month}${day}_${hours}${minutes}${seconds}`;
}
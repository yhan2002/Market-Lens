import type { Metadata } from "next";
import { IBM_Plex_Mono, Noto_Sans_KR } from "next/font/google";
import "./globals.css";

const sans = Noto_Sans_KR({ variable: "--font-sans", subsets: ["latin"], weight: ["400", "500", "600", "700", "800"] });
const mono = IBM_Plex_Mono({ variable: "--font-mono", subsets: ["latin"], weight: ["400", "500", "600"] });

export const metadata: Metadata = {
  title: "Market Lens Risk Management",
  description: "Market Health, HMM, fast-risk overlay and sector shadow dashboard",
  openGraph: {
    title: "Market Lens Risk Management",
    description: "시장 국면·위험 오버레이·권고 Long 비중·재진입 shadow 대시보드",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="ko"><body className={`${sans.variable} ${mono.variable}`}>{children}</body></html>;
}

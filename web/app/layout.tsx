import type { Metadata } from "next";
import { Public_Sans } from "next/font/google";
import { TriangleAlert } from "lucide-react";
import { Header } from "@/components/Header";
import { WorkbenchProvider } from "@/lib/workbench";
import "./globals.css";
import "./form.css";

const sans = Public_Sans({ subsets: ["latin"], weight: ["400", "500", "600", "700", "800"], display: "swap" });

export const metadata: Metadata = {
  title: "TrialMatch · criterion screening",
  description: "Screen a patient note against clinical-trial criteria with cited evidence. Educational prototype.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={sans.className}>
      <body>
        <div className="banner" role="note">
          <TriangleAlert size={15} aria-hidden="true" />
          <span className="long">Educational screening prototype — not a clinical enrollment decision.</span>
          <span className="short">Educational prototype — not a clinical decision.</span>
        </div>
        <WorkbenchProvider>
          <Header />
          <main className="page">{children}</main>
        </WorkbenchProvider>
      </body>
    </html>
  );
}

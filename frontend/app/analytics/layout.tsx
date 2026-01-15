"use client";

import React from "react";
import { DashboardLayout } from "@/components/layouts/dashboard-layout";

export default function AnalyticsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <DashboardLayout>{children}</DashboardLayout>;
}

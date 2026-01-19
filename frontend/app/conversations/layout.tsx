"use client";

import React from "react";
import { DashboardLayout } from "@/components/layouts/dashboard-layout";

export default function ConversationsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <DashboardLayout>{children}</DashboardLayout>;
}

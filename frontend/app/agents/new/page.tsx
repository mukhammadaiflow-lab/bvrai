"use client";

import { AgentWizard } from "@/components/agents";

export default function NewAgentPage() {
  return (
    <div className="py-6">
      <AgentWizard mode="create" />
    </div>
  );
}

import { NotAvailableYet } from "@/components/ui/States";

export default function AutoMLRunsPage() {
  return (
    <NotAvailableYet
      feature="AutoML runs"
      description="Automated model training with Optuna tuning, leaderboards, and SHAP explainability arrives in the AutoML phase of the roadmap."
    />
  );
}

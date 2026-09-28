"use client";

import { useState } from "react";
import { Dialog } from "@/components/dialog";
import { EmptyState } from "@/components/empty-state";
import { ErrorNotice } from "@/components/error-notice";
import { PageHeader } from "@/components/page-header";
import { StatusBadge } from "@/components/status-badge";
import { useToast } from "@/components/toast";
import { ApiError } from "@/lib/api/client";
import { useDatasets, useRegisterDataset, useValidateDataset } from "@/lib/api/queries";
import { formatCount, formatDate } from "@/lib/format";

export function DatasetsPage() {
  const datasets = useDatasets();
  const register = useRegisterDataset();
  const validate = useValidateDataset();
  const toast = useToast();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [source, setSource] = useState<"synthetic" | "upload">("synthetic");
  const [uri, setUri] = useState("");
  async function submit() {
    try {
      await register.mutateAsync({ name, source, uri });
      toast("Dataset registered");
      setOpen(false);
      setName("");
      setUri("");
    } catch (error) {
      toast(error instanceof ApiError ? error.message : "The dataset could not be registered.");
    }
  }
  async function onValidate(id: string) {
    try {
      const dataset = await validate.mutateAsync(id);
      toast(dataset.status === "valid" ? "Dataset valid" : "Dataset invalid");
    } catch (error) {
      toast(error instanceof ApiError ? error.message : "Validation could not be completed.");
    }
  }
  if (datasets.isPending) {
    return (
      <main className="page">
        <PageHeader title="Datasets" description="Registered sales history and whether it passed validation." />
        <p className="skeleton-block" aria-hidden="true" />
      </main>
    );
  }
  if (datasets.error) {
    return (
      <main className="page">
        <PageHeader title="Datasets" description="Registered sales history and whether it passed validation." />
        <ErrorNotice error={datasets.error} onRetry={() => void datasets.refetch()} />
      </main>
    );
  }
  return (
    <main className="page">
      <PageHeader
        title="Datasets"
        description="Registered sales history and whether it passed validation."
        actions={
          <button type="button" className="btn btn-primary" onClick={() => setOpen(true)}>
            Register dataset
          </button>
        }
      />
      {(datasets.data?.items.length ?? 0) === 0 ? (
        <EmptyState
          title="No datasets registered"
          body="Register the sample dataset or point at your own sales history."
        />
      ) : (
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Version</th>
                <th>Status</th>
                <th className="numeric">Rows</th>
                <th>Last date</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {datasets.data?.items.map((dataset) => (
                <tr key={dataset.id}>
                  <td>{dataset.name}</td>
                  <td className="mono">{dataset.version}</td>
                  <td>
                    <StatusBadge kind="dataset" value={dataset.status} />
                  </td>
                  <td className="numeric">{formatCount(dataset.row_count)}</td>
                  <td>{formatDate(dataset.date_max)}</td>
                  <td>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      disabled={validate.isPending}
                      onClick={() => void onValidate(dataset.id)}
                    >
                      Validate dataset
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <Dialog title="Register dataset" open={open} onClose={() => setOpen(false)}>
        <label className="field">
          Name
          <input value={name} onChange={(event) => setName(event.target.value)} />
        </label>
        <label className="field">
          Source
          <select value={source} onChange={(event) => setSource(event.target.value === "upload" ? "upload" : "synthetic")}>
            <option value="synthetic">Synthetic</option>
            <option value="upload">Upload</option>
          </select>
        </label>
        <label className="field">
          Directory
          <input value={uri} onChange={(event) => setUri(event.target.value)} placeholder="Path on this machine" />
        </label>
        <div className="dialog-actions">
          <button type="button" className="btn btn-secondary" data-cancel onClick={() => setOpen(false)}>
            Cancel
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={register.isPending || name.trim() === "" || uri.trim() === ""}
            onClick={() => void submit()}
          >
            {register.isPending ? "Registering…" : "Register dataset"}
          </button>
        </div>
      </Dialog>
    </main>
  );
}

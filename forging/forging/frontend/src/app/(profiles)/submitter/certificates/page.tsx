import Link from "next/link";

export default function SubmitterCertificatesPage() {
  return (
    <div className="flex min-h-screen flex-col bg-background-light text-text-main">
      <header className="border-b border-border-color bg-white px-6 py-6 shadow-subtle lg:px-10">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-[0.22em] text-primary">
              Certificates
            </p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight">
              Submission Certificates
            </h1>
            <p className="mt-2 max-w-2xl text-sm font-medium text-muted">
              Track certificate availability and issued verification records for
              submitted documents.
            </p>
          </div>
          <Link
            className="rounded-full bg-primary px-5 py-3 text-sm font-bold text-white transition-opacity hover:opacity-90"
            href="/submitter/my-submissions"
          >
            View My Submissions
          </Link>
        </div>
      </header>

      <main className="flex-1 p-6 lg:p-8">
        <section className="rounded-[28px] border border-border-color bg-white p-6 shadow-subtle lg:p-8">
          <div className="rounded-[24px] border border-dashed border-border-color bg-background-light px-6 py-16 text-center">
            <p className="text-xs font-bold uppercase tracking-[0.18em] text-muted">
              Certificates
            </p>
            <h2 className="mt-3 text-2xl font-bold tracking-tight">
              No certificates issued yet
            </h2>
            <p className="mt-3 text-sm font-medium text-muted">
              Certificate records will appear here when a submission completes
              the verification workflow and an issuance artifact is generated.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}

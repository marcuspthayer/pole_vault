import Link from 'next/link';

export default function LandingPage() {
  return (
    <main className="min-h-screen bg-gray-950 text-white flex flex-col">
      <nav className="px-8 py-4 flex justify-between items-center border-b border-gray-800">
        <span className="text-xl font-bold">VaultSense</span>
        <div className="flex gap-4">
          <Link href="/auth/login" className="text-gray-400 hover:text-white text-sm">
            Sign in
          </Link>
          <Link
            href="/auth/signup"
            className="px-4 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-700 text-sm font-medium"
          >
            Get started
          </Link>
        </div>
      </nav>

      <div className="flex-1 flex flex-col items-center justify-center text-center px-4 space-y-6">
        <h1 className="text-5xl font-bold tracking-tight">
          AI-powered pole vault<br />biomechanics analysis
        </h1>
        <p className="text-gray-400 text-lg max-w-xl">
          Upload a vault video. Get instant feedback on approach velocity, pole bend,
          hip height, and cadence — compared to elite benchmarks.
        </p>
        <Link
          href="/auth/signup"
          className="px-8 py-3 rounded-xl bg-blue-600 hover:bg-blue-700 text-lg font-semibold"
        >
          Analyze your vault
        </Link>
      </div>
    </main>
  );
}

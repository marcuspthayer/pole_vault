import Link from 'next/link';
import { MediaPlaceholder } from './MediaPlaceholder';

export function Hero() {
  return (
    <section className="max-w-4xl mx-auto px-6 py-20 md:py-28 text-center space-y-8">
      <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
        AI-powered pole vault
        <br />
        biomechanics analysis
      </h1>
      <p className="text-gray-400 text-lg max-w-2xl mx-auto leading-relaxed">
        Upload a vault video and get instant biomechanics feedback. Our two-pass
        AI pipeline tracks 33 skeletal landmarks, measures pole bend, and
        computes 8+ performance metrics — all compared to elite benchmarks.
      </p>
      <Link
        href="/auth/signup"
        className="inline-block px-8 py-3 rounded-xl bg-blue-600 hover:bg-blue-700 text-lg font-semibold transition-colors"
      >
        Analyze your vault
      </Link>
      <div className="max-w-3xl mx-auto pt-4">
        <MediaPlaceholder
          label="Skeleton + pole tracking demo"
          sublabel="Coming soon"
        />
      </div>
    </section>
  );
}

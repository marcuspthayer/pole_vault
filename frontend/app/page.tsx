import Link from 'next/link';
import { Navbar } from '@/components/landing/Navbar';
import { Hero } from '@/components/landing/Hero';
import { HowItWorks } from '@/components/landing/HowItWorks';
import { MetricsShowcase } from '@/components/landing/MetricsShowcase';
import { FeatureHighlights } from '@/components/landing/FeatureHighlights';
import { FilmingGuide } from '@/components/landing/FilmingGuide';
import { Footer } from '@/components/landing/Footer';

export default function LandingPage() {
  return (
    <main className="min-h-screen bg-gray-950 text-white flex flex-col">
      <Navbar />
      <Hero />
      <HowItWorks />
      <MetricsShowcase />
      <FeatureHighlights />
      <FilmingGuide />

      {/* Final CTA */}
      <section className="text-center px-6 py-16 md:py-20">
        <h2 className="text-3xl md:text-4xl font-bold">
          Ready to analyze your vault?
        </h2>
        <p className="text-gray-400 text-lg mt-3">
          Upload a video and get instant biomechanics feedback.
        </p>
        <Link
          href="/auth/signup"
          className="inline-block mt-8 px-8 py-3 rounded-xl bg-blue-600 hover:bg-blue-700 text-lg font-semibold transition-colors"
        >
          Get started free
        </Link>
      </section>

      <Footer />
    </main>
  );
}

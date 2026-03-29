import { createClient } from '@/lib/supabase/server';
import { redirect } from 'next/navigation';
import Link from 'next/link';

export default async function DashboardPage() {
  const supabase = await createClient();
  const { data: { user } } = await supabase.auth.getUser();

  if (!user) redirect('/auth/login');

  const { data: profile } = await supabase
    .from('profiles')
    .select('*')
    .eq('id', user.id)
    .single();

  return (
    <main className="min-h-screen bg-gray-950 text-white">
      <nav className="px-8 py-4 flex justify-between items-center border-b border-gray-800">
        <span className="text-xl font-bold">VaultSense</span>
        <div className="flex items-center gap-4">
          <span className="text-gray-400 text-sm">{user.email}</span>
          <form action="/auth/signout" method="post">
            <button className="text-gray-400 hover:text-white text-sm">Sign out</button>
          </form>
        </div>
      </nav>

      <div className="max-w-4xl mx-auto px-8 py-12 space-y-8">
        <div>
          <h1 className="text-3xl font-bold">
            Welcome{profile?.full_name ? `, ${profile.full_name}` : ''}
          </h1>
          <p className="text-gray-400 mt-1 capitalize">{profile?.role ?? 'athlete'} account</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Link
            href="/analyze"
            className="p-6 rounded-xl bg-gray-900 border border-gray-800 hover:border-blue-500 transition-colors space-y-2"
          >
            <h2 className="text-lg font-semibold">Analyze a vault</h2>
            <p className="text-gray-400 text-sm">Upload a video and get biomechanics feedback.</p>
          </Link>

          <div className="p-6 rounded-xl bg-gray-900 border border-gray-800 space-y-2 opacity-50">
            <h2 className="text-lg font-semibold">Session history</h2>
            <p className="text-gray-400 text-sm">View past analyses and track improvement over time.</p>
            <span className="text-xs text-blue-400">Coming soon</span>
          </div>
        </div>
      </div>
    </main>
  );
}

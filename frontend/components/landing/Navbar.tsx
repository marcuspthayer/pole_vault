import Link from 'next/link';

export function Navbar() {
  return (
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
  );
}

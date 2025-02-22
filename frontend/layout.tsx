import './globals.css'
import { Quicksand } from 'next/font/google'

const quicksand = Quicksand({ 
  subsets: ['latin'],
  display: 'swap',
})

export const metadata = {
  title: 'Story Sage - Make Your Story Come to Life',
  description: 'Transform your stories into captivating audiobooks with AI-powered narration',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={quicksand.className}>{children}</body>
    </html>
  )
}

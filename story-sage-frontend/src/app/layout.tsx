'use client'
import './globals.css'
import { Quicksand } from 'next/font/google'
import Image from 'next/image'
import { usePathname } from 'next/navigation'

const quicksand = Quicksand({ 
  subsets: ['latin'],
  display: 'swap',
})

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const pathname = usePathname()
  const showLogo = pathname !== '/upload'

  return (
    <html lang="en">
      <body className={quicksand.className}>
        {/* Two decorative sage leaves, randomly positioned */}
        <div className="sage-leaf-corner left-[5%] top-[20%]" /> {/* Left side leaf */}
        <div className="sage-leaf-corner right-[8%] bottom-[35%] rotate-[145deg]" /> {/* Right side leaf */}
        
        {/* Logo Header - Only show on non-upload pages */}
        {showLogo && (
          <div className="logo-header mt-2">
            <Image
              src="/logo.png"
              alt="Story Sage Logo"
              width={200}
              height={150}
              className="logo-image"
              priority
            />
          </div>
        )}
        
        <div className={showLogo ? "-mt-8" : ""}>
          {children}
        </div>
      </body>
    </html>
  )
}

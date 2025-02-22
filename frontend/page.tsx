'use client'

import Link from 'next/link'
import { motion } from 'framer-motion'

export default function Home() {
  return (
    <main className="min-h-screen relative overflow-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 bg-[url('/patterns/natural-paper.png')] opacity-10 z-0" />
      
      {/* Hero Section */}
      <div className="relative z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-16 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h1 className="hero-text mb-6">
              Story Sage
            </h1>
            <p className="subtitle-text max-w-3xl mx-auto">
              Make Your Story Come to Life
            </p>
            
            {/* Nature-inspired decorative element */}
            <div className="my-12 flex justify-center">
              <div className="h-0.5 w-24 bg-gradient-to-r from-emerald-500 to-teal-400" />
            </div>
            
            {/* CTA Button */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3, duration: 0.8 }}
              className="mt-8"
            >
              <Link href="/upload" className="btn-primary">
                Get Started
              </Link>
            </motion.div>
          </motion.div>
          
          {/* Features Section */}
          <div className="mt-24 grid grid-cols-1 md:grid-cols-3 gap-12">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 * index, duration: 0.8 }}
                className="p-6 rounded-2xl bg-white/50 backdrop-blur-sm shadow-xl hover:shadow-2xl transition-shadow"
              >
                <h3 className="text-xl font-semibold text-gray-800 mb-3">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </main>
  )
}

const features = [
  {
    title: "AI-Powered Narration",
    description: "Transform your text into lifelike audio with advanced AI technology."
  },
  {
    title: "Natural Voice",
    description: "Experience storytelling with incredibly natural and expressive voices."
  },
  {
    title: "Easy to Use",
    description: "Simply upload your text and let Story Sage bring your stories to life."
  }
]

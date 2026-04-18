import { motion } from "framer-motion"; import { Button } from "@/components/ui/button"; import { Card, CardContent } from "@/components/ui/card";

export default function SafeDostLanding() { return ( <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-black text-white"> {/* Hero Section */} <section className="text-center py-20 px-6"> <motion.h1 initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }} className="text-5xl font-bold mb-6" > SafeDost.AI </motion.h1> <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }} className="text-xl text-gray-300 mb-8" > Your AI-powered safety companion for real-time guidance </motion.p> <Button className="rounded-2xl px-6 py-3 text-lg"> Try Demo </Button> </section>

{/* Features */}
  <section className="grid md:grid-cols-3 gap-6 px-8 py-16">
    {["Real-Time Guidance", "Verified Knowledge", "AI Intelligence"].map((feature, i) => (
      <motion.div
        key={i}
        initial={{ opacity: 0, y: 30 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ delay: i * 0.2 }}
      >
        <Card className="bg-gray-800 border-none rounded-2xl shadow-lg">
          <CardContent className="p-6">
            <h3 className="text-xl font-semibold mb-2">{feature}</h3>
            <p className="text-gray-400">
              Powerful feature to ensure safety and reliability in real-time situations.
            </p>
          </CardContent>
        </Card>
      </motion.div>
    ))}
  </section>

  {/* How It Works */}
  <section className="text-center py-16 px-6">
    <h2 className="text-3xl font-bold mb-10">How It Works</h2>
    <div className="flex flex-col md:flex-row justify-center gap-6">
      {["User Query", "AI Processing", "Actionable Response"].map((step, i) => (
        <motion.div
          key={i}
          initial={{ opacity: 0, scale: 0.9 }}
          whileInView={{ opacity: 1, scale: 1 }}
          transition={{ delay: i * 0.2 }}
          className="bg-gray-800 p-6 rounded-2xl"
        >
          {step}
        </motion.div>
      ))}
    </div>
  </section>

  {/* CTA */}
  <section className="text-center py-20">
    <h2 className="text-3xl font-bold mb-6">
      Empowering Safety Through AI
    </h2>
    <Button className="rounded-2xl px-6 py-3 text-lg">
      Get Started
    </Button>
  </section>

  {/* Footer */}
  <footer className="text-center py-6 text-gray-500">
    © {new Date().getFullYear()} SafeDost.AI
  </footer>
</div>

); }

import {
  HelpCircle,
  Book,
  MessageCircle,
  Mail,
  ExternalLink,
  Search,
  ChevronRight,
  FileText,
  Video,
  Users,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

const faqs = [
  {
    question: "How accurate is Verifixia's detection?",
    answer:
      "Verifixia maintains a 98.7% accuracy rate in detecting deepfakes across various manipulation types including face swaps, lip syncing, and full synthesis. Our model is continuously trained on the latest deepfake techniques.",
  },
  {
    question: "What video formats are supported?",
    answer:
      "We support all major video formats including MP4, AVI, MOV, WebM, and MKV. Live streaming analysis supports RTMP, HLS, and WebRTC protocols.",
  },
  {
    question: "How do I integrate Verifixia AI with my existing systems?",
    answer:
      "Verifixia AI provides RESTful APIs for easy integration. Visit your Profile > API Keys section to generate authentication tokens. Our documentation includes SDKs for Python, JavaScript, and Go.",
  },
  {
    question: "What happens when a deepfake is detected?",
    answer:
      "When a deepfake is detected, the system immediately flags the content, logs the incident with full metadata, and triggers configured alerts. You can set up automatic actions like quarantine, block, or manual review.",
  },
  {
    question: "How is my data handled and stored?",
    answer:
      "All data is encrypted at rest and in transit using AES-256 encryption. Video frames are processed in memory and discarded after analysis unless you enable logging. We are SOC 2 Type II compliant.",
  },
];

const resources = [
  {
    title: "Getting Started Guide",
    description: "Learn the basics of Verifixia AI",
    icon: Book,
    type: "Documentation",
  },
  {
    title: "API Reference",
    description: "Complete API documentation",
    icon: FileText,
    type: "Documentation",
  },
  {
    title: "Video Tutorials",
    description: "Step-by-step video guides",
    icon: Video,
    type: "Video",
  },
  {
    title: "Community Forum",
    description: "Connect with other users",
    icon: Users,
    type: "Community",
  },
];

export const Support = () => {
  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Support Center</h1>
        <p className="text-muted-foreground">
          Get help and find answers to common questions
        </p>
      </div>

      {/* Search */}
      <Card className="glass-card border-border/50">
        <CardContent className="pt-6">
          <div className="relative max-w-2xl mx-auto">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
            <Input
              placeholder="Search for help articles, tutorials, or FAQs..."
              className="pl-12 h-12 text-lg"
            />
          </div>
        </CardContent>
      </Card>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="glass-card border-border/50 hover:border-primary/30 transition-colors cursor-pointer">
          <CardContent className="pt-6 flex flex-col items-center text-center">
            <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
              <MessageCircle className="w-6 h-6 text-primary" />
            </div>
            <h3 className="font-semibold mb-1">Live Chat</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Chat with our support team
            </p>
            <Button variant="outline" size="sm">
              Start Chat
            </Button>
          </CardContent>
        </Card>

        <Card className="glass-card border-border/50 hover:border-primary/30 transition-colors cursor-pointer">
          <CardContent className="pt-6 flex flex-col items-center text-center">
            <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
              <Mail className="w-6 h-6 text-primary" />
            </div>
            <h3 className="font-semibold mb-1">Email Support</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Get help via email
            </p>
            <Button variant="outline" size="sm">
              Send Email
            </Button>
          </CardContent>
        </Card>

        <Card className="glass-card border-border/50 hover:border-primary/30 transition-colors cursor-pointer">
          <CardContent className="pt-6 flex flex-col items-center text-center">
            <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
              <Book className="w-6 h-6 text-primary" />
            </div>
            <h3 className="font-semibold mb-1">Documentation</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Browse our knowledge base
            </p>
            <Button variant="outline" size="sm">
              View Docs
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Resources */}
      <Card className="glass-card border-border/50">
        <CardHeader>
          <CardTitle>Resources</CardTitle>
          <CardDescription>Helpful guides and documentation</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {resources.map((resource) => (
              <div
                key={resource.title}
                className="flex items-center gap-4 p-4 rounded-lg border border-border/50 bg-secondary/20 hover:bg-secondary/40 transition-colors cursor-pointer group"
              >
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <resource.icon className="w-5 h-5 text-primary" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium">{resource.title}</p>
                  <p className="text-sm text-muted-foreground">
                    {resource.description}
                  </p>
                </div>
                <ChevronRight className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* FAQs */}
      <Card className="glass-card border-border/50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <HelpCircle className="w-5 h-5" />
            Frequently Asked Questions
          </CardTitle>
          <CardDescription>Quick answers to common questions</CardDescription>
        </CardHeader>
        <CardContent>
          <Accordion type="single" collapsible className="w-full">
            {faqs.map((faq, index) => (
              <AccordionItem key={index} value={`item-${index}`}>
                <AccordionTrigger className="text-left">
                  {faq.question}
                </AccordionTrigger>
                <AccordionContent className="text-muted-foreground">
                  {faq.answer}
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </CardContent>
      </Card>

      {/* Contact */}
      <Card className="glass-card border-primary/20 bg-primary/5">
        <CardContent className="pt-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div>
              <h3 className="text-lg font-semibold">Still need help?</h3>
              <p className="text-muted-foreground">
                Our enterprise support team is available 24/7
              </p>
            </div>
            <Button>
              Contact Enterprise Support
              <ExternalLink className="w-4 h-4 ml-2" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Support;

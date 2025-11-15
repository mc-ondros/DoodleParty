import React from 'react';
import { CheckCircleIcon } from '../../constants';

const PricingCard: React.FC<{ plan: string; price: string; description: string; features: string[]; isFeatured?: boolean; }> = ({ plan, price, description, features, isFeatured = false }) => (
  <div className={`border rounded-xl p-8 flex flex-col ${isFeatured ? 'border-green-500 bg-zinc-900' : 'border-zinc-700 bg-zinc-800/50'}`}>
    <h3 className="text-xl font-bold text-white">{plan}</h3>
    <p className="mt-2 text-zinc-400">{description}</p>
    <div className="mt-6">
      <span className="text-5xl font-extrabold text-white">{price}</span>
      { plan !== 'Hobby' && <span className="text-zinc-400">/ month</span>}
    </div>
    <ul className="mt-8 space-y-4 text-zinc-300 flex-grow">
      {features.map((feature, index) => (
        <li key={index} className="flex items-start">
          <CheckCircleIcon className="w-5 h-5 mr-3 mt-1 text-green-500 flex-shrink-0" />
          <span>{feature}</span>
        </li>
      ))}
    </ul>
    <button className={`mt-10 w-full py-3 font-semibold rounded-lg transition-colors ${isFeatured ? 'bg-green-500 text-black hover:bg-green-600' : 'bg-zinc-700 text-white hover:bg-zinc-600'}`}>
      {plan === 'Enterprise' ? 'Contact Sales' : 'Get Started'}
    </button>
  </div>
);

const PricingPage: React.FC = () => {
  return (
    <div className="bg-black text-white min-h-full p-10">
      <div className="text-center max-w-3xl mx-auto">
        <h1 className="text-5xl font-extrabold">Find a plan to power your creativity</h1>
        <p className="mt-4 text-lg text-zinc-400">
          Whether you're doodling for fun, running a community event, or deploying at scale, we have a plan that's right for you.
        </p>
      </div>
      <div className="mt-16 max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8">
        <PricingCard
          plan="Hobby"
          price="Free"
          description="For individuals and small groups just starting out."
          features={['Up to 10 concurrent users','Classic Canvas mode','Basic AI moderation','Community support']}
        />
        <PricingCard
          plan="Pro"
          price="$25"
          description="For event organizers, streamers, and growing communities."
          features={['Up to 100 concurrent users','All game modes included','Advanced AI moderation','Custom branding options','Priority support','Analytics dashboard']}
          isFeatured
        />
        <PricingCard
          plan="Enterprise"
          price="Custom"
          description="For large-scale deployments and unique requirements."
          features={['Unlimited concurrent users','On-premises RPi4 deployment option','Dedicated infrastructure','Custom LLM integration','24/7 premium support','Service Level Agreement (SLA)']}
        />
      </div>
    </div>
  );
};

export default PricingPage;

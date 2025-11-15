import React from 'react';
import type { Page } from '../../App';
import { CheckCircleIcon, HelpCircleIcon, ChevronRightIcon } from '../../constants';

interface RightSidebarProps {
  currentPage: Page;
}


const GettingStarted: React.FC = () => (
  <div className="bg-zinc-900 p-6 rounded-lg border border-zinc-800">
    <h3 className="text-lg font-bold mb-1">Welcome to DoodleParty 👋</h3>
    <p className="text-sm text-zinc-400 mb-4">Let's follow these simple steps to get you ready!</p>
    
    <div className="w-full bg-zinc-700 rounded-full h-1.5 mb-6">
      <div className="bg-green-500 h-1.5 rounded-full" style={{width: '33%'}}></div>
    </div>
    
    <div className="space-y-4">
      <div className="opacity-50">
        <h4 className="font-semibold flex items-center"><CheckCircleIcon className="w-4 h-4 mr-2 text-green-500" /> Verify your email</h4>
        <p className="text-sm text-zinc-400 pl-6">Click on a verification link that we've sent to your email address.</p>
      </div>
       <div>
        <h4 className="font-semibold">Customize your profile</h4>
        <p className="text-sm text-zinc-400">Choose an avatar and a cool username to stand out in the community.</p>
        <button className="mt-2 w-full bg-white text-black font-semibold py-2 rounded-md text-sm hover:bg-zinc-200 transition-colors">EDIT PROFILE</button>
      </div>
      <div className="opacity-50">
        <h4 className="font-semibold">Join a canvas!</h4>
        <p className="text-sm text-zinc-400">Time to get creative. Jump into the live canvas and make your first doodle!</p>
      </div>
    </div>
  </div>
);

const RightSidebar: React.FC<RightSidebarProps> = ({ currentPage }) => {
    const renderContent = () => {
        switch (currentPage) {
            case 'explore':
                return <GettingStarted />;
            default:
                return null;
        }
    };
    
    const content = renderContent();
    if (!content) return null;

  return (
    <div className="w-80 bg-black flex-shrink-0 p-6 border-l border-zinc-800 flex flex-col">
       <div className="flex items-center text-sm font-semibold mb-6 h-5">
            { content &&
                <>
                    <ChevronRightIcon className="w-5 h-5 rotate-180 mr-2"/>
                    {currentPage === 'explore' && 'Getting Started'}
                </>
            }
       </div>
       <div className="flex-grow overflow-y-auto pr-2 -mr-2">
        {content}
       </div>
       <button className="mt-auto flex items-center justify-center w-full bg-zinc-800 py-2.5 rounded-md hover:bg-zinc-700 transition-colors flex-shrink-0">
            <HelpCircleIcon className="w-5 h-5 mr-2" />
            <span className="font-semibold">HAVE A QUESTION?</span>
       </button>
    </div>
  );
};

export default RightSidebar;
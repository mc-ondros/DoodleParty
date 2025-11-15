
import React, { useState, useEffect } from 'react';

const calculateTimeLeft = () => {
    const difference = +new Date("2024-11-06T22:00:00") - +new Date();
    let timeLeft = {
        days: 0,
        hours: 0,
        minutes: 0,
        seconds: 0
    };

    if (difference > 0) {
        timeLeft = {
            days: Math.floor(difference / (1000 * 60 * 60 * 24)),
            hours: Math.floor((difference / (1000 * 60 * 60)) % 24),
            minutes: Math.floor((difference / 1000 / 60) % 60),
            seconds: Math.floor((difference / 1000) % 60)
        };
    }
    
    // For demonstration, let's set a fixed time
    timeLeft.days = 3;
    timeLeft.hours = 19;
    timeLeft.minutes = 54;

    return timeLeft;
};

const CountdownTimer: React.FC = () => {
    const [timeLeft] = useState(calculateTimeLeft());
    const [seconds, setSeconds] = useState(54);

    useEffect(() => {
        const timer = setTimeout(() => {
            setSeconds(seconds > 0 ? seconds - 1 : 59);
        }, 1000);

        return () => clearTimeout(timer);
    }, [seconds]);

    return (
        <div className="flex items-end space-x-2 md:space-x-4">
            <div className="flex items-baseline">
                <span className="text-2xl md:text-4xl lg:text-5xl font-black">{String(timeLeft.days).padStart(2, '0')}</span>
                <span className="text-[10px] md:text-xs lg:text-sm font-bold ml-0.5 md:ml-1">DAYS</span>
            </div>
             <div className="text-2xl md:text-4xl lg:text-5xl font-black pb-0.5 md:pb-1">:</div>
            <div className="flex items-baseline">
                <span className="text-2xl md:text-4xl lg:text-5xl font-black">{String(timeLeft.hours).padStart(2, '0')}</span>
                <span className="text-[10px] md:text-xs lg:text-sm font-bold ml-0.5 md:ml-1">HRS</span>
            </div>
             <div className="text-2xl md:text-4xl lg:text-5xl font-black pb-0.5 md:pb-1">:</div>
            <div className="flex items-baseline">
                <span className="text-2xl md:text-4xl lg:text-5xl font-black">{String(timeLeft.minutes).padStart(2, '0')}</span>
                <span className="text-[10px] md:text-xs lg:text-sm font-bold ml-0.5 md:ml-1">MIN</span>
            </div>
             <div className="text-2xl md:text-4xl lg:text-5xl font-black pb-0.5 md:pb-1 hidden sm:block">:</div>
            <div className="flex items-baseline hidden sm:flex">
                <span className="text-2xl md:text-4xl lg:text-5xl font-black">{String(seconds).padStart(2, '0')}</span>
                <span className="text-[10px] md:text-xs lg:text-sm font-bold ml-0.5 md:ml-1">SEC</span>
            </div>
        </div>
    );
};

export default CountdownTimer;

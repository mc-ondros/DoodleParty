
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
        <div className="flex items-end space-x-4">
            <div className="flex items-baseline">
                <span className="text-5xl font-black">{String(timeLeft.days).padStart(2, '0')}</span>
                <span className="text-sm font-bold ml-1">DAYS</span>
            </div>
             <div className="text-5xl font-black pb-1">:</div>
            <div className="flex items-baseline">
                <span className="text-5xl font-black">{String(timeLeft.hours).padStart(2, '0')}</span>
                <span className="text-sm font-bold ml-1">HOURS</span>
            </div>
             <div className="text-5xl font-black pb-1">:</div>
            <div className="flex items-baseline">
                <span className="text-5xl font-black">{String(timeLeft.minutes).padStart(2, '0')}</span>
                <span className="text-sm font-bold ml-1">MINUTES</span>
            </div>
             <div className="text-5xl font-black pb-1">:</div>
            <div className="flex items-baseline">
                <span className="text-5xl font-black">{String(seconds).padStart(2, '0')}</span>
            </div>
        </div>
    );
};

export default CountdownTimer;

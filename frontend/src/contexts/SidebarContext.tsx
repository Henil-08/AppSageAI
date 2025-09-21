'use client';

import { createContext, useState, useContext, ReactNode } from 'react';

// Define the shape of the context data
interface SidebarContextType {
  sidebarOpen: boolean;
  setSidebarOpen: (isOpen: boolean) => void;
}

// Create the context with a default value of null
const SidebarContext = createContext<SidebarContextType | null>(null);

// Create a Provider component that will hold the state
export function SidebarProvider({ children }: { children: ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <SidebarContext.Provider value={{ sidebarOpen, setSidebarOpen }}>
      {children}
    </SidebarContext.Provider>
  );
}

// Create a custom hook for easy access to the context
export function useSidebar() {
  const context = useContext(SidebarContext);
  if (!context) {
    throw new Error('useSidebar must be used within a SidebarProvider');
  }
  return context;
}
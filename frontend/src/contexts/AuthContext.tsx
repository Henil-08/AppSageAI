'use client';

import React, { createContext, useContext, useEffect, useState } from 'react';
import { User } from 'firebase/auth';
import { auth, signInWithGoogle, signOut, getIdToken } from '../lib/firebase';
import { useRouter } from 'next/navigation';
import toast from 'react-hot-toast';

interface AuthContextType {
  user: User | null;
  loading: boolean;
  signIn: () => Promise<void>;
  signOut: () => Promise<void>;
  getToken: () => Promise<string | null>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: React.ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged(async (user) => {
      setUser(user);
      
      if (user) {
        // User is signed in
        const token = await user.getIdToken();
        localStorage.setItem('firebase_token', token);
        
        // Verify with backend
        try {
          const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/auth/verify`, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json',
            },
          });
          
          if (response.ok) {
            const userData = await response.json();
            console.log('User verified:', userData);
          }
        } catch (error) {
          console.error('Error verifying user:', error);
        }
      } else {
        // User is signed out
        localStorage.removeItem('firebase_token');
      }
      
      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  const signIn = async () => {
    try {
      setLoading(true);
      const result = await signInWithGoogle();
      toast.success(`Welcome, ${result.user.displayName}!`);
      router.push('/dashboard');
    } catch (error: any) {
      console.error('Sign in error:', error);
      toast.error(error.message || 'Failed to sign in');
    } finally {
      setLoading(false);
    }
  };

  const signOutUser = async () => {
    try {
      setLoading(true);
      await signOut();
      toast.success('Signed out successfully');
      router.push('/');
    } catch (error: any) {
      console.error('Sign out error:', error);
      toast.error('Failed to sign out');
    } finally {
      setLoading(false);
    }
  };

  const getToken = async () => {
    return await getIdToken();
  };

  const value = {
    user,
    loading,
    signIn,
    signOut: signOutUser,
    getToken,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
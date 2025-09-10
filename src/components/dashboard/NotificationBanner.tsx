import React, { useState } from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Info, X } from 'lucide-react';

export const NotificationBanner: React.FC = () => {
  const [isVisible, setIsVisible] = useState(true);

  if (!isVisible) return null;

  return (
    <Alert className="border-primary/20 bg-primary/5">
      <Info className="h-4 w-4 text-primary" />
      <AlertDescription className="flex items-center justify-between">
        <span className="text-foreground">
          <strong>Notice about filing of Foreign Exchange Management Application Proceedings are Alleged filing, 2000:</strong>
          <br />
          Whether any online form been issued under sub rule 3 of foreign Exchange Management (Application Proceedings are Alleged filing, 2000.
        </span>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsVisible(false)}
          className="h-6 w-6 p-0 hover:bg-primary/10"
        >
          <X className="h-4 w-4" />
        </Button>
      </AlertDescription>
    </Alert>
  );
};
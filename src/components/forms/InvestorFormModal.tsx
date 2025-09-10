import React from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { CustomSelect, SelectOption } from '@/components/dashboard/CustomSelect';
import { Formik, Form, Field, ErrorMessage, FieldArray } from 'formik';
import * as Yup from 'yup';
import { X, Plus, Trash2, UserPlus } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface InvestorFormModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const investorValidationSchema = Yup.object({
  name: Yup.string().required('Name is required'),
  address: Yup.string().required('Address is required'),
  contribution: Yup.string().required('Contribution type is required'),
  residentialStatus: Yup.string().required('Residential status is required'),
  reportNeeded: Yup.string().required('Report needed status is required'),
  amount: Yup.number().required('Amount is required').min(1, 'Amount must be greater than 0'),
  percentage: Yup.number().required('Percentage is required').min(0).max(100, 'Percentage cannot exceed 100'),
});

const validationSchema = Yup.object({
  investors: Yup.array().of(investorValidationSchema).min(1, 'At least one investor is required')
});

const contributionOptions: SelectOption[] = [
  { value: 'contractual', label: 'Contractual' },
  { value: 'economic_club', label: 'Economic Club' },
  { value: 'equity', label: 'Equity Investment' },
  { value: 'debt', label: 'Debt Investment' }
];

const residentialStatusOptions: SelectOption[] = [
  { value: 'resident', label: 'Resident' },
  { value: 'non_resident', label: 'Non Resident' },
  { value: 'nri', label: 'NRI' },
  { value: 'foreign', label: 'Foreign National' }
];

const reportNeededOptions: SelectOption[] = [
  { value: 'yes', label: 'Yes' },
  { value: 'no', label: 'No' },
  { value: 'non_resident', label: 'Non Resident' },
  { value: 'pending', label: 'Pending' }
];

const initialInvestor = {
  name: '',
  address: '',
  contribution: '',
  residentialStatus: '',
  reportNeeded: '',
  amount: '',
  percentage: ''
};

export const InvestorFormModal: React.FC<InvestorFormModalProps> = ({ isOpen, onClose }) => {
  const { toast } = useToast();

  const initialValues = {
    investors: [initialInvestor]
  };

  const handleSubmit = async (values: typeof initialValues, { setSubmitting }: any) => {
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      toast({
        title: "Investor Data Submitted",
        description: `Successfully added ${values.investors.length} investor(s) to the database.`,
      });
      
      console.log('Investor data submitted:', values);
      onClose();
    } catch (error) {
      toast({
        title: "Submission Failed",
        description: "There was an error submitting the investor data. Please try again.",
        variant: "destructive",
      });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
        <DialogHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
          <DialogTitle className="text-xl font-semibold">Add Investor Information</DialogTitle>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </DialogHeader>

        <Formik
          initialValues={initialValues}
          validationSchema={validationSchema}
          onSubmit={handleSubmit}
        >
          {({ values, setFieldValue, isSubmitting, errors }) => (
            <Form className="space-y-6">
              <FieldArray name="investors">
                {({ push, remove }) => (
                  <div className="space-y-6">
                    <div className="flex items-center justify-between">
                      <h3 className="text-lg font-medium text-foreground">Investor Details</h3>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() => push(initialInvestor)}
                      >
                        <Plus className="h-4 w-4 mr-2" />
                        Add Investor
                      </Button>
                    </div>

                    {values.investors.map((investor, index) => (
                      <div key={index} className="p-4 border border-border rounded-lg space-y-4">
                        <div className="flex items-center justify-between">
                          <h4 className="font-medium text-foreground">Investor #{index + 1}</h4>
                          {values.investors.length > 1 && (
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              onClick={() => remove(index)}
                              className="text-destructive hover:text-destructive"
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          )}
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                          <div className="space-y-2">
                            <Label htmlFor={`investors.${index}.name`}>Name *</Label>
                            <Field
                              as={Input}
                              id={`investors.${index}.name`}
                              name={`investors.${index}.name`}
                              placeholder="Enter investor name"
                            />
                            <ErrorMessage name={`investors.${index}.name`} component="div" className="text-xs text-destructive" />
                          </div>

                          <div className="space-y-2">
                            <Label htmlFor={`investors.${index}.address`}>Address *</Label>
                            <Field
                              as={Input}
                              id={`investors.${index}.address`}
                              name={`investors.${index}.address`}
                              placeholder="Enter address"
                            />
                            <ErrorMessage name={`investors.${index}.address`} component="div" className="text-xs text-destructive" />
                          </div>

                          <div className="space-y-2">
                            <Label>Contribution *</Label>
                            <CustomSelect
                              options={contributionOptions}
                              value={investor.contribution}
                              onChange={(value) => setFieldValue(`investors.${index}.contribution`, value)}
                              placeholder="Select contribution type"
                            />
                            <ErrorMessage name={`investors.${index}.contribution`} component="div" className="text-xs text-destructive" />
                          </div>

                          <div className="space-y-2">
                            <Label>Residential Status *</Label>
                            <CustomSelect
                              options={residentialStatusOptions}
                              value={investor.residentialStatus}
                              onChange={(value) => setFieldValue(`investors.${index}.residentialStatus`, value)}
                              placeholder="Select status"
                            />
                            <ErrorMessage name={`investors.${index}.residentialStatus`} component="div" className="text-xs text-destructive" />
                          </div>

                          <div className="space-y-2">
                            <Label>Report Needed *</Label>
                            <CustomSelect
                              options={reportNeededOptions}
                              value={investor.reportNeeded}
                              onChange={(value) => setFieldValue(`investors.${index}.reportNeeded`, value)}
                              placeholder="Select option"
                            />
                            <ErrorMessage name={`investors.${index}.reportNeeded`} component="div" className="text-xs text-destructive" />
                          </div>

                          <div className="space-y-2">
                            <Label htmlFor={`investors.${index}.amount`}>Amount (₹) *</Label>
                            <Field
                              as={Input}
                              id={`investors.${index}.amount`}
                              name={`investors.${index}.amount`}
                              type="number"
                              placeholder="0"
                            />
                            <ErrorMessage name={`investors.${index}.amount`} component="div" className="text-xs text-destructive" />
                          </div>

                          <div className="space-y-2">
                            <Label htmlFor={`investors.${index}.percentage`}>Percentage (%) *</Label>
                            <Field
                              as={Input}
                              id={`investors.${index}.percentage`}
                              name={`investors.${index}.percentage`}
                              type="number"
                              min="0"
                              max="100"
                              placeholder="0"
                            />
                            <ErrorMessage name={`investors.${index}.percentage`} component="div" className="text-xs text-destructive" />
                          </div>
                        </div>
                      </div>
                    ))}

                    {typeof errors.investors === 'string' && (
                      <div className="text-xs text-destructive">{errors.investors}</div>
                    )}
                  </div>
                )}
              </FieldArray>

              {/* Action Buttons */}
              <div className="flex justify-end space-x-3 pt-6 border-t">
                <Button type="button" variant="outline" onClick={onClose}>
                  Cancel
                </Button>
                <Button type="submit" disabled={isSubmitting}>
                  {isSubmitting ? (
                    <>
                      <div className="h-4 w-4 border-2 border-primary-foreground border-t-transparent rounded-full animate-spin mr-2" />
                      Saving...
                    </>
                  ) : (
                    <>
                      <UserPlus className="h-4 w-4 mr-2" />
                      Save Investors
                    </>
                  )}
                </Button>
              </div>
            </Form>
          )}
        </Formik>
      </DialogContent>
    </Dialog>
  );
};
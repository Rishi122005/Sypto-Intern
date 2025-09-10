import React from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { CustomSelect, SelectOption } from '@/components/dashboard/CustomSelect';
import { Formik, Form, Field, ErrorMessage } from 'formik';
import * as Yup from 'yup';
import { X, FileText } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface TaxFormModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const validationSchema = Yup.object({
  applicantName: Yup.string()
    .required('Applicant name is required')
    .min(2, 'Name must be at least 2 characters'),
  panNumber: Yup.string()
    .required('PAN number is required')
    .matches(/^[A-Z]{5}[0-9]{4}[A-Z]{1}$/, 'Invalid PAN format'),
  applicationForm: Yup.string()
    .required('Application form type is required'),
  residentialStatus: Yup.string()
    .required('Residential status is required'),
  investmentAmount: Yup.number()
    .required('Investment amount is required')
    .min(1, 'Amount must be greater than 0'),
  investmentPurpose: Yup.string()
    .required('Investment purpose is required'),
  bankDetails: Yup.string()
    .required('Bank details are required'),
  additionalInfo: Yup.string()
});

const applicationFormOptions: SelectOption[] = [
  { value: 'form_a', label: 'Form A - Foreign Exchange Application' },
  { value: 'form_fc_gpr', label: 'Form FC-GPR - General Permission Route' },
  { value: 'form_fc_trs', label: 'Form FC-TRS - Transfer of Shares' },
  { value: 'form_diip', label: 'Form DIIP - Direct Investment' }
];

const residentialStatusOptions: SelectOption[] = [
  { value: 'resident', label: 'Resident Indian' },
  { value: 'nri', label: 'Non-Resident Indian' },
  { value: 'foreign', label: 'Foreign National' },
  { value: 'pio', label: 'Person of Indian Origin' }
];

const investmentPurposeOptions: SelectOption[] = [
  { value: 'business', label: 'Business Investment' },
  { value: 'property', label: 'Property Investment' },
  { value: 'securities', label: 'Securities Investment' },
  { value: 'other', label: 'Other Investment' }
];

export const TaxFormModal: React.FC<TaxFormModalProps> = ({ isOpen, onClose }) => {
  const { toast } = useToast();

  const initialValues = {
    applicantName: '',
    panNumber: '',
    applicationForm: '',
    residentialStatus: '',
    investmentAmount: '',
    investmentPurpose: '',
    bankDetails: '',
    additionalInfo: ''
  };

  const handleSubmit = async (values: typeof initialValues, { setSubmitting }: any) => {
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      toast({
        title: "Form Submitted Successfully",
        description: "Your tax application has been submitted for processing.",
      });
      
      console.log('Form submitted:', values);
      onClose();
    } catch (error) {
      toast({
        title: "Submission Failed",
        description: "There was an error submitting your form. Please try again.",
        variant: "destructive",
      });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
          <DialogTitle className="text-xl font-semibold">Tax & RBI Application Form</DialogTitle>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </DialogHeader>

        <Formik
          initialValues={initialValues}
          validationSchema={validationSchema}
          onSubmit={handleSubmit}
        >
          {({ values, setFieldValue, isSubmitting, errors, touched }) => (
            <Form className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Applicant Information */}
                <div className="space-y-4">
                  <h3 className="text-lg font-medium text-foreground">Applicant Information</h3>
                  
                  <div className="space-y-2">
                    <Label htmlFor="applicantName">Applicant Name *</Label>
                    <Field
                      as={Input}
                      id="applicantName"
                      name="applicantName"
                      placeholder="Enter full name"
                      className={errors.applicantName && touched.applicantName ? 'border-destructive' : ''}
                    />
                    <ErrorMessage name="applicantName" component="div" className="text-xs text-destructive" />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="panNumber">PAN Number *</Label>
                    <Field
                      as={Input}
                      id="panNumber"
                      name="panNumber"
                      placeholder="ABCDE1234F"
                      className={errors.panNumber && touched.panNumber ? 'border-destructive' : ''}
                    />
                    <ErrorMessage name="panNumber" component="div" className="text-xs text-destructive" />
                  </div>

                  <div className="space-y-2">
                    <Label>Residential Status *</Label>
                    <CustomSelect
                      options={residentialStatusOptions}
                      value={values.residentialStatus}
                      onChange={(value) => setFieldValue('residentialStatus', value)}
                      placeholder="Select residential status"
                      className={errors.residentialStatus && touched.residentialStatus ? 'border-destructive' : ''}
                    />
                    <ErrorMessage name="residentialStatus" component="div" className="text-xs text-destructive" />
                  </div>
                </div>

                {/* Application Details */}
                <div className="space-y-4">
                  <h3 className="text-lg font-medium text-foreground">Application Details</h3>
                  
                  <div className="space-y-2">
                    <Label>Application Form Type *</Label>
                    <CustomSelect
                      options={applicationFormOptions}
                      value={values.applicationForm}
                      onChange={(value) => setFieldValue('applicationForm', value)}
                      placeholder="Select application form"
                      className={errors.applicationForm && touched.applicationForm ? 'border-destructive' : ''}
                    />
                    <ErrorMessage name="applicationForm" component="div" className="text-xs text-destructive" />
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="investmentAmount">Investment Amount (₹) *</Label>
                    <Field
                      as={Input}
                      id="investmentAmount"
                      name="investmentAmount"
                      type="number"
                      placeholder="0"
                      className={errors.investmentAmount && touched.investmentAmount ? 'border-destructive' : ''}
                    />
                    <ErrorMessage name="investmentAmount" component="div" className="text-xs text-destructive" />
                  </div>

                  <div className="space-y-2">
                    <Label>Investment Purpose *</Label>
                    <CustomSelect
                      options={investmentPurposeOptions}
                      value={values.investmentPurpose}
                      onChange={(value) => setFieldValue('investmentPurpose', value)}
                      placeholder="Select investment purpose"
                      className={errors.investmentPurpose && touched.investmentPurpose ? 'border-destructive' : ''}
                    />
                    <ErrorMessage name="investmentPurpose" component="div" className="text-xs text-destructive" />
                  </div>
                </div>
              </div>

              {/* Full width fields */}
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="bankDetails">Bank Details *</Label>
                  <Field
                    as={Textarea}
                    id="bankDetails"
                    name="bankDetails"
                    placeholder="Enter bank name, account number, IFSC code, etc."
                    rows={3}
                    className={errors.bankDetails && touched.bankDetails ? 'border-destructive' : ''}
                  />
                  <ErrorMessage name="bankDetails" component="div" className="text-xs text-destructive" />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="additionalInfo">Additional Information</Label>
                  <Field
                    as={Textarea}
                    id="additionalInfo"
                    name="additionalInfo"
                    placeholder="Any additional information or special circumstances"
                    rows={3}
                  />
                  <ErrorMessage name="additionalInfo" component="div" className="text-xs text-destructive" />
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex justify-end space-x-3 pt-6 border-t">
                <Button type="button" variant="outline" onClick={onClose}>
                  Cancel
                </Button>
                <Button type="submit" disabled={isSubmitting}>
                  {isSubmitting ? (
                    <>
                      <div className="h-4 w-4 border-2 border-primary-foreground border-t-transparent rounded-full animate-spin mr-2" />
                      Submitting...
                    </>
                  ) : (
                    <>
                      <FileText className="h-4 w-4 mr-2" />
                      Submit Application
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
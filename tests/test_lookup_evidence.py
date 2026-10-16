import unittest
from rag_contest.lookup_evidence import lookup_evidence
from rag_contest.retrieval import WordCodec, Retriever
from test_contest import document


class LookupTests(unittest.TestCase):
    def excerpt(self, text, question, budget=500):
        return lookup_evidence(document(1,text),WordCodec(),budget,question)

    def test_table_keeps_header_and_matching_row(self):
        text='Table 1. Fleet\nBoat Class Built Hull\nRobin 3x 2001 Teak\nOwl 2x 2002 Pine\nTable 2. Moisture\nTimber Reading\nTeak 19'
        got=self.excerpt(text,'What class is the boat made of teak?')
        self.assertIn('Boat Class Built Hull',got)
        self.assertIn('Robin 3x 2001 Teak',got)
        self.assertNotIn('Owl',got)
        self.assertNotIn('Moisture',got)

    def test_ambiguous_material_falls_back(self):
        self.assertEqual(self.excerpt('Table 1. Fleet\nBoat Class Hull\nA 3x Teak\nB 2x Teak',
                                     'What class is the boat made of teak?'),'')

    def test_person_keeps_wrapped_role_not_other_section(self):
        got=self.excerpt('1. Witnesses\nWitnesses: Ana Luna (commissioning\nengineer, Acme), Ben (manager).\n2. Checks\nAna Luna checked the sprinkler.',
                         "What's Ana Luna's job title?")
        self.assertIn('commissioning engineer',got)
        self.assertNotIn('sprinkler',got)

    def test_definition_excludes_example(self):
        got=self.excerpt('2. Method\nT = f * m\nwhere f = 1.4 the dynamic factor, m is mass\nin kg.\nWorked example: f*m = 350.',
                         'What value of the dynamic factor is used?')
        self.assertIn('1.4',got)
        self.assertIn('in kg.',got)
        self.assertNotIn('350',got)

    def test_policy_keeps_exceptions(self):
        got=self.excerpt('3. Fines paid by the employer\nExpired inspection certificate.\nUnless the employee hid the notice, the employer pays.\n4. Renewal\nLicence renewal fee 60.',
                         'Who pays for an expired inspection certificate?')
        self.assertIn('Unless',got)
        self.assertIn('paid by the employer',got)
        self.assertNotIn('renewal fee',got)

    def test_budget_and_unsupported_lookup_fall_back(self):
        self.assertEqual(self.excerpt('A long document', 'What is the name?'),'')
        self.assertEqual(self.excerpt('Table 1. Fleet\nBoat Class Hull\nA 3x Teak',
                                     'What class is the boat made of teak?',1),'')

    def test_v4_fallback_preserves_policy(self):
        d=document(1,'1. Warranty\nFittings last 2 years. Structures last 7 years.')
        r=Retriever([d],WordCodec())
        candidate=r.search('warranty')[0]
        args=(candidate,WordCodec(),200,'full','Is the warranty seven years?')
        self.assertEqual(r.evidence(*args,strategy='focused_v4'),r.evidence(*args,strategy='focused_v3'))

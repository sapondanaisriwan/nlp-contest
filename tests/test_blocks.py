import unittest
from rag_contest.blocks import complete_evidence, source_units
from rag_contest.retrieval import WordCodec


class BlockTests(unittest.TestCase):
    def test_focused_keeps_urgent_exception(self):
        doc=dict(document_id='1',pages=[dict(page=1,text='1. Normal hours\nOutside normal office hours wait until Monday.\n2. Mandatory handover\nSmoke or fire must be transferred.\n3. Urgent safety\nUrgent messages get a callback within 5 minutes, even outside office hours.')])
        result=complete_evidence(doc,WordCodec(),150,'Smoke was seen outside normal office hours. Should a response wait until Monday?',focused=True)
        self.assertIn('5 minutes',result)
        self.assertIn('Smoke or fire',result)

    def test_focused_count_list_survives_brand_noise(self):
        doc=dict(document_id='1',pages=[dict(page=1,text='What is Acme Alert?\n'+'Acme alert service '*20+'\nWhat do codes mean?\nGREEN go. AMBER wait. RED stop. STORM shelter.\nHow to pay?\nAcme alert costs 20.')])
        result=complete_evidence(doc,WordCodec(),100,'Are there 4 code in Acme alert?',focused=True)
        for word in ('GREEN','AMBER','RED','STORM'):self.assertIn(word,result)

    def test_focused_exact_row_beats_repeated_query_noise(self):
        doc=dict(document_id='1',pages=[dict(page=1,text='Table 1. Survey\nSite Distance (m)\nAB-02 75\n2. Background\n'+'AB-02 survey history '*60)])
        result=complete_evidence(doc,WordCodec(),50,'How far away is AB-02?',focused=True)
        self.assertIn('Site Distance (m)\nAB-02 75',result)

    def test_focused_does_not_fill_remaining_budget_with_weak_sections(self):
        doc=dict(document_id='1',pages=[dict(page=1,text='1. Name\nThe app name is LiftGo.\n2. Contacts\nMedia contact for the app.\n3. History\nHistory of the app.\n4. Office\nOffice for the app.')])
        result=complete_evidence(doc,WordCodec(),300,'What is the name of the app?',focused=True)
        self.assertIn('LiftGo',result)
        self.assertLessEqual(result.count('[Document'),2)

    def test_singular_query_ranks_plural_heading_over_brand_noise(self):
        doc=dict(document_id='1',pages=[dict(page=1,text='What is Acme Alert?\n'+'Acme alert service '*20+'\nWhat do codes mean?\nGREEN go. AMBER wait. RED stop. STORM shelter.\nHow to pay?\nAcme alert costs 20.')])
        result=complete_evidence(doc,WordCodec(),35,'Are there 4 code in Acme alert?')
        for word in ('GREEN','AMBER','RED','STORM'):self.assertIn(word,result)

    def test_numbered_faq(self):
        doc=dict(document_id='1',pages=[dict(page=1,text='1. Which category applies?\nCategory B: wrong network.\n2. How to appeal?\n'+'appeal '*100)])
        result=complete_evidence(doc,WordCodec(),40,'Which category is wrong network?')
        self.assertIn('Category B',result)
        self.assertNotIn('How to appeal',result)

    def test_year_and_numbered_sentences_do_not_split_unit(self):
        doc = dict(document_id='1', pages=[dict(page=1,text='1. Policy\n2026. The owner checks parts.\n1. Bring a receipt.\n2. Bring the parts.\n2. Fees\nCost is 20.')])
        units=source_units(doc)
        self.assertEqual(len(units),2)
        self.assertIn('2. Bring the parts.',units[0]['text'])

    def test_faq_list_cross_page_is_kept_complete(self):
        doc = dict(document_id='1', pages=[dict(page=1, text='1. Service\nWhat codes exist?\nGREEN: clear\nAMBER: wait'),
                                          dict(page=2, text='RED: stop\nSTORM: shelter\nHow to cancel?\n'+'cancellation '*100)])
        result = complete_evidence(doc, WordCodec(), 55, 'Are there four codes?')
        for word in ('GREEN', 'AMBER', 'RED', 'STORM'): self.assertIn(word, result)
        self.assertNotIn('cancellation', result)
        self.assertLessEqual(len(WordCodec().encode(result)),55)

    def test_oversize_unit_never_clipped(self):
        doc = dict(document_id='1', pages=[dict(page=1, text='What codes exist?\n'+'code '*100)])
        self.assertEqual(complete_evidence(doc, WordCodec(), 20, 'codes'), '')

    def test_table_columns_stay_with_row(self):
        doc = dict(document_id='1', pages=[dict(page=1, text='Table 1. Fleet\nClass Material\n2x Mahogany\n2. Other\nNoise')])
        result = complete_evidence(doc, WordCodec(), 30, 'What class uses Mahogany?')
        self.assertIn('Class Material\n2x Mahogany',result)

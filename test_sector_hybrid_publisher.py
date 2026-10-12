import unittest
from unittest.mock import patch
import sector_hybrid_publisher as publisher


class PublisherTests(unittest.TestCase):
    def test_dates_and_failure_are_not_old_signals(self):
        with patch.object(publisher, 'build_snapshot', return_value={'as_of':'2026-09-18','latest_price_date':'2026-09-21','status':'READY','rows':[]}):
            value = publisher.public_snapshot('unused.db','2026-09-21')
        self.assertEqual(value['as_of'],'2026-09-21')
        self.assertEqual(value['decision_date'],'2026-09-18')
        with patch.object(publisher, 'build_snapshot', side_effect=ValueError('secret not to be exposed')):
            value = publisher.public_snapshot('unused.db','2026-09-21')
        self.assertEqual(value['status'],'ERROR')
        self.assertEqual(value['rows'],[])
        self.assertIsNone(value['decision_date'])
        self.assertNotIn('secret',str(value))


if __name__=='__main__': unittest.main()

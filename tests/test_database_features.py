import unittest

from modules import database


class DatabaseFeatureTests(unittest.TestCase):
    def test_delete_helpers_exist_and_remove_rows(self):
        self.assertTrue(hasattr(database, "delete_saved_place"))
        self.assertTrue(hasattr(database, "delete_document"))
        self.assertTrue(hasattr(database, "delete_reminder"))
        self.assertTrue(hasattr(database, "delete_expense"))

        database.initialize_database()
        destination = "TestCity"

        database.save_place(destination, "Old Fort", "attraction", "Historic site")
        database.delete_saved_place(destination, "Old Fort")
        self.assertEqual(database.saved_place_rows(destination), [])

        database.add_document(destination, "Passport", "Passport", "2027-01-01", "Primary travel doc")
        database.delete_document(destination, "Passport")
        self.assertEqual(database.document_rows(destination), [])

        database.add_reminder(destination, "Check route", "2027-01-02", "08:30")
        database.delete_reminder(destination, "Check route")
        self.assertEqual(database.reminder_rows(destination), [])

        database.add_expense(destination, "Food", "Test meal", 500, "Me")
        expense_id = database.expense_rows(destination)[0]["id"]
        database.delete_expense(expense_id)
        self.assertEqual(database.expense_rows(destination), [])


if __name__ == "__main__":
    unittest.main()

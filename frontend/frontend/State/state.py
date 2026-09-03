import httpx
import reflex as rx


API_URL="http://127.0.0.1:8001/"


# Global client for connection pooling
api_client = httpx.AsyncClient(base_url=API_URL)

class NotebookState(rx.State):
    notebooks: list[dict] = []
    loading: bool = False
    creating: bool = False

    async def load_notebooks(self):
       self.loading =True
       try:
           response = await api_client.get("notebooks")
           response.raise_for_status()
           notebooks = response.json()
           for nb in notebooks:
               if "_id" in nb:
                   nb["id"] = nb.pop("_id")
           self.notebooks = notebooks

       except Exception as e:
           print(f"Error: {e}")
       finally:
           self.loading = False

    async def create_notebook(self, form_data: dict):
        self.creating = True
        try:
            response = await api_client.post("notebooks", json=form_data)
            response.raise_for_status()
            new_notebook = response.json()
            if "_id" in new_notebook:
                new_notebook["id"] = new_notebook.pop("_id")
            self.notebooks.append(new_notebook)

        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.creating = False

        return rx.redirect("/")

    async def delete_notebook(self, notebook_id: str):

        try:
            response = await api_client.delete(f"notebooks/{notebook_id}")
            response.raise_for_status()

            self.notebooks = [
                notebook
                for notebook in self.notebooks
                if notebook["id"] != notebook_id
            ]

        except Exception as e:
            print("Failed to delete notebook:", e)


    async def rename_notebook(
        self,
        notebook_id: str,
        new_title: str,
    ):

        if not new_title.strip():
            return

        payload = {
            "title": new_title.strip()
        }

        try:
            response = await api_client.patch(
                f"notebooks/{notebook_id}",
                json=payload,
            )
            response.raise_for_status()

            updated = response.json()
            if "_id" in updated:
                updated["id"] = updated.pop("_id")

            self.notebooks = [
                updated
                if notebook["id"] == notebook_id
                else notebook
                for notebook in self.notebooks
            ]

        except Exception as e:
            print("Failed to rename notebook:", e)
from typing import List


class HTMLUtil:
    @staticmethod
    def add_concept_to_html(concept:List[str], html:str):
        newline = '\n'
        return html.replace(
"""
<!--
-->
""",
f"""
<!--
{newline.join(concept)}
-->
"""
        )

    @staticmethod
    def add_script_to_html(script:List[str], html:str):
        newline = '\n'
        return html.replace(
"""
<script>
</script>
""",
f"""
<script>
{newline.join(script)}
</script>
"""
        )
    
    @staticmethod
    def save_html(html, save_path):
        with open(save_path, "w", encoding= "utf-8") as f:
            f.write(html)
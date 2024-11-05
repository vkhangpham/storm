import json
import re
import argparse

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def load_markdown(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def save_markdown(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

def extract_citations(text):
    return list(set(re.findall(r'\[(\d+)\]', text)))

def map_citations_to_resources(citations, instance_dump):
    citation_map = {}
    info_uuid_to_info_dict = instance_dump['knowledge_base']['info_uuid_to_info_dict']
    
    for citation in citations:
        uuid = int(citation)
        if str(uuid) in info_uuid_to_info_dict:
            resource = info_uuid_to_info_dict[str(uuid)]
            citation_map[citation] = {
                'url': resource['url'],
                'title': resource['title'],
                'description': resource['description']
            }
        else:
            citation_map[citation] = "Resource not found"
    
    return citation_map

def format_references(citation_map):
    references = []
    for citation, resource in sorted(citation_map.items(), key=lambda x: int(x[0])):
        if isinstance(resource, dict):
            references.append(f"\n<a id='ref-{citation}'></a>{citation}. [{resource['title']}]({resource['url']})")
        else:
            references.append(f"\n<a id='ref-{citation}'></a>{citation}. {resource}")
    return "\n".join(references)

def clean_content(content):
    content = re.sub(r'(\[(\d+)\](?:\[\2\])?|\(#ref-\d+\))+', lambda m: f"[{m.group(2)}]" if m.group(2) else "", content)
    content = re.sub(r'\n## References\n[\s\S]*$', '', content)
    return content.strip()

def add_citation_links(content, citations):
    for citation in citations:
        pattern = r'\[' + re.escape(citation) + r'\](?!\[' + re.escape(citation) + r'\])'
        replacement = f"[{citation}](#ref-{citation})"
        content = re.sub(pattern, replacement, content)
    return content

def main():
    parser = argparse.ArgumentParser(description="Map citations in a report to resources in an instance dump.")
    parser.add_argument("report_path", help="Path to the report markdown file")
    parser.add_argument("instance_dump_path", help="Path to the instance dump JSON file")
    args = parser.parse_args()

    report_content = load_markdown(args.report_path)
    instance_dump = load_json(args.instance_dump_path)

    report_content = clean_content(report_content)

    citations = extract_citations(report_content)
    citation_map = map_citations_to_resources(citations, instance_dump)

    references = format_references(citation_map)
    
    updated_content = add_citation_links(report_content, citations)
    updated_report = f"{updated_content}\n\n## References\n\n{references}"
    
    save_markdown(args.report_path, updated_report)
    
    print("References have been added to the report and citations have been linked.")

if __name__ == "__main__":
    main()

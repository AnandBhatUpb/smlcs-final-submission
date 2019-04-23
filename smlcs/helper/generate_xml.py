import pandas as pd
import xml.etree.cElementTree as ET


def indent(elem, level=0):
  i = "\n" + level*"  "
  if len(elem):
    if not elem.text or not elem.text.strip():
      elem.text = i + "  "
    if not elem.tail or not elem.tail.strip():
      elem.tail = i
    for elem in elem:
      indent(elem, level+1)
    if not elem.tail or not elem.tail.strip():
      elem.tail = i
  else:
    if level and (not elem.tail or not elem.tail.strip()):
      elem.tail = i


predictions = pd.read_csv('../dataset/new_method.csv')
programs = list(predictions['programs'].values)
configurations = list(predictions['configurations'].values)

for i in range(len(programs)):
    root = ET.Element("benchmark", timelimit="300", tool="cpachecker")
    doc = ET.SubElement(root, "tasks", name="runSet")

    ET.SubElement(doc, "include").text = "../experiment_programs/"+str(programs[i])
    ET.SubElement(doc, "propertyfile").text = "../ALL.prp"

    option1 = ET.SubElement(root, "option", name="-heap").text = "10000M"
    option2 = ET.SubElement(root, "option", name="-"+str(configurations[i]))
    option3 = ET.SubElement(root, "option", name="-disable-java-assertions")
    option4 = ET.SubElement(root, "option", name="-spec").text = "default"
    option5 = ET.SubElement(root, "option", name="-preprocess")
    option6 = ET.SubElement(root, "option", name="-skipRecursion")

    runtimedef = ET.SubElement(root, "rundefinition", name=str(configurations[i]))
    ET.SubElement(runtimedef, "option", name="-noout")

    indent(root)

    doctype = '<!DOCTYPE benchmark PUBLIC \'+//IDN sosy-lab.org//DTD BenchExec benchmark 1.0//EN\' ' \
              '\'http://www.sosy-lab.org/benchexec/benchmark-1.0.dtd\'>'

    with open('../../new_method/run'+str((i+1))+'.xml', 'wb') as f:
        f.write('<?xml version="1.0" ?>\n<!DOCTYPE benchmark\n  PUBLIC \'+//IDN sosy-lab.org//DTD BenchExec benchmark 1.0//EN\'\n  \'http://www.sosy-lab.org/benchexec/benchmark-1.0.dtd\'>\n'.encode('utf8'))
        ET.ElementTree(root).write(f)


import streamlit as st
from pinecone import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate

# Configuración de la página
st.set_page_config(page_title="Consulta de Base de Datos Vectorial", layout="wide")
st.title("🔍 Sistema de Consulta Inteligente con Pinecone")

# Función para obtener índices de Pinecone
def get_pinecone_indexes(api_key):
    try:
        pc = Pinecone(api_key=api_key)
        current_indexes = pc.list_indexes().names()
        return list(current_indexes)
    except Exception as e:
        st.error(f"Error al obtener índices: {str(e)}")
        return []

# Función para limpiar estados
def clear_all_states():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.experimental_rerun()

# Inicialización de estados
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

# Sidebar para configuración
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    
    # Campo para OpenAI API Key
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Introduce tu API key de OpenAI"
    )
    
    # Campo para Pinecone API Key
    pinecone_api_key = st.text_input(
        "Pinecone API Key",
        type="password",
        help="Introduce tu API key de Pinecone"
    )
    
    # Selector de modelo LLM
    llm_model = st.selectbox(
        "Modelo LLM",
        options=["gpt-3.5-turbo", "gpt-4o-mini"],
        help="Selecciona el modelo de lenguaje a utilizar"
    )
    
    # Temperatura del modelo
    temperature = st.slider(
        "Temperatura",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        help="Controla la creatividad de las respuestas"
    )
    
    # Botones de control
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Actualizar"):
            st.experimental_rerun()
    with col2:
        if st.button("🗑️ Limpiar"):
            clear_all_states()
    
    # Verificar conexión con Pinecone y mostrar índices
    if pinecone_api_key:
        try:
            st.markdown("### 📊 Estado")
            available_indexes = get_pinecone_indexes(pinecone_api_key)
            
            if available_indexes:
                st.success("✅ Conectado a Pinecone")
                
                # Selector de índice
                selected_index = st.selectbox(
                    "Selecciona un índice",
                    options=available_indexes
                )
                
                # Mostrar información del índice seleccionado
                if selected_index:
                    pc = Pinecone(api_key=pinecone_api_key)
                    index = pc.Index(selected_index)
                    stats = index.describe_index_stats()
                    
                    # Mostrar estadísticas básicas
                    st.markdown("#### 📈 Estadísticas")
                    total_vectors = stats.get('total_vector_count', 0)
                    st.metric("Total de vectores", total_vectors)
                    
                    # Mostrar namespaces disponibles
                    if 'namespaces' in stats:
                        st.markdown("#### 🏷️ Namespaces")
                        namespaces = list(stats['namespaces'].keys())
                        if namespaces:
                            selected_namespace = st.selectbox(
                                "Selecciona un namespace",
                                options=namespaces
                            )
                            st.session_state.namespace = selected_namespace
                        else:
                            st.info("No hay namespaces disponibles")
                            st.session_state.namespace = ""
            else:
                st.warning("⚠️ No hay índices disponibles")
                selected_index = None
                
        except Exception as e:
            st.error(f"❌ Error de conexión: {str(e)}")
            selected_index = None
    else:
        selected_index = None

def get_enhanced_response(query, context_results, llm):
    """Genera una respuesta mejorada usando el modelo LLM."""
    # Template para el prompt
    template = """
    Actúa como un asistente experto y útil. Basándote en la siguiente información y pregunta,
    genera una respuesta, breve, clara, precisa y bien estructurada, si no hay contexto, di que no tienes contexto.

    Pregunta: {query}

    Contexto relevante:
    {context}

    Por favor, proporciona una respuesta que:
    1. Sea relevante y directa
    2. Esté bien estructurada y sea fácil de entender
    3. Incluya ejemplos o referencias cuando sea apropiado
    4. Mantenga un tono profesional pero amigable

    Respuesta:
    """

    # Preparar el contexto
    context_texts = []
    for match in context_results.matches:
        if 'text' in match.metadata:
            context_texts.append(f"- {match.metadata['text']}")
    
    context = "\n".join(context_texts)
    
    # Crear el prompt
    prompt = ChatPromptTemplate.from_template(template)
    
    # Generar la respuesta
    messages = prompt.format_messages(
        query=query,
        context=context
    )
    
    response = llm.invoke(messages)
    return response.content

def query_pinecone(query_text, namespace, k=5):
    try:
        # Inicializar embeddings y LLM
        embedding_model = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-ada-002"
        )
        
        llm = ChatOpenAI(
            temperature=temperature,
            model_name=llm_model,
            openai_api_key=openai_api_key
        )
        
        # Generar embedding para la consulta
        query_embedding = embedding_model.embed_query(query_text)
        
        # Inicializar Pinecone
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(selected_index)
        
        # Realizar búsqueda
        results = index.query(
            vector=query_embedding,
            namespace=namespace,
            top_k=k,
            include_metadata=True
        )
        
        # Generar respuesta mejorada
        enhanced_response = get_enhanced_response(query_text, results, llm)
        
        return results, enhanced_response
        
    except Exception as e:
        st.error(f"Error en la consulta: {str(e)}")
        return None, None

# Interface principal
if openai_api_key and pinecone_api_key and selected_index:
    st.markdown("### 🔍 Realizar Consulta")
    
    # Parámetros de búsqueda
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("💭 ¿Qué deseas consultar?")
    with col2:
        k = st.number_input("Número de resultados", min_value=1, max_value=10, value=5)
    
    # Botón de búsqueda
    if st.button("🔍 Buscar"):
        if query:
            with st.spinner("🔄 Buscando y procesando..."):
                results, enhanced_response = query_pinecone(
                    query,
                    namespace=getattr(st.session_state, 'namespace', ''),
                    k=k
                )
                
                if results and hasattr(results, 'matches'):
                    # Mostrar respuesta mejorada
                    st.markdown("### 🤖 Respuesta Generada:")
                    st.write(enhanced_response)
                    
                    # Mostrar fuentes
                    st.markdown("### 📚 Fuentes Consultadas:")
                    
                    for i, match in enumerate(results.matches, 1):
                        score = match.score
                        similarity = round((1 - (1 - score)) * 100, 2)
                        
                        with st.expander(f"📍 Fuente {i} - Similitud: {similarity}%"):
                            if 'text' in match.metadata:
                                st.write(match.metadata['text'])
                            else:
                                st.write("No se encontró texto en los metadatos")
                            
                            # Mostrar metadatos adicionales
                            other_metadata = {k:v for k,v in match.metadata.items() if k != 'text'}
                            if other_metadata:
                                st.markdown("##### Metadatos adicionales:")
                                st.json(other_metadata)
                else:
                    st.warning("No se encontraron resultados")
        else:
            st.warning("⚠️ Por favor, ingresa una consulta")
else:
    st.info("👈 Por favor, configura las credenciales en el panel lateral para comenzar")

# Información en el sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### ℹ️ Sobre esta aplicación")
    st.write("""
    Esta aplicación te permite realizar consultas semánticas mejoradas con IA en bases de datos
    vectoriales existentes en Pinecone.
    
    Características:
    - Conexión directa a índices de Pinecone
    - Búsqueda semántica con OpenAI
    - Procesamiento con LLM para mejorar respuestas
    - Soporte para múltiples namespaces
    - Visualización de similitud y fuentes
    """)
